# QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? 

**Title (ZH)**: QuestBench: LLMs在推理任务中能否问出正确的问题以获取信息？ 

**Authors**: Belinda Z. Li, Been Kim, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22674)  

**Abstract**: Recently, a large amount of work has focused on improving large language models' (LLMs') performance on reasoning benchmarks such as math and logic. However, past work has largely assumed that tasks are well-defined. In the real world, queries to LLMs are often underspecified, only solvable through acquiring missing information. We formalize this as a constraint satisfaction problem (CSP) with missing variable assignments. Using a special case of this formalism where only one necessary variable assignment is missing, we can rigorously evaluate an LLM's ability to identify the minimal necessary question to ask and quantify axes of difficulty levels for each problem. We present QuestBench, a set of underspecified reasoning tasks solvable by asking at most one question, which includes: (1) Logic-Q: Logical reasoning tasks with one missing proposition, (2) Planning-Q: PDDL planning problems with initial states that are partially-observed, (3) GSM-Q: Human-annotated grade school math problems with one missing variable assignment, and (4) GSME-Q: a version of GSM-Q where word problems are translated into equations by human annotators. The LLM is tasked with selecting the correct clarification question(s) from a list of options. While state-of-the-art models excel at GSM-Q and GSME-Q, their accuracy is only 40-50% on Logic-Q and Planning-Q. Analysis demonstrates that the ability to solve well-specified reasoning problems may not be sufficient for success on our benchmark: models have difficulty identifying the right question to ask, even when they can solve the fully specified version of the problem. Furthermore, in the Planning-Q domain, LLMs tend not to hedge, even when explicitly presented with the option to predict ``not sure.'' This highlights the need for deeper investigation into models' information acquisition capabilities. 

**Abstract (ZH)**: 最近，大量研究集中在提高大型语言模型（LLMs）在数学和逻辑等推理基准上的性能。然而，以往的工作主要假设任务是明确定义的。在现实世界中，对LLMs的查询通常不明确，需要通过获取缺失信息来解决。我们将这种情形形式化为一种带有缺失变量赋值的约束满足问题（CSP）。通过这种方法的一个特殊情况，即仅缺少一个必要变量赋值，我们可以严格评估LLMs识别最小必要询问的能力，并量化每个问题的难度轴。我们提出了QuestBench，这是一个包含最多可提出一个询问以解决的任务集，包括：（1）Logic-Q：缺少一个命题的逻辑推理任务，（2）Planning-Q：初始状态部分未观察到的PDDL规划问题，（3）GSM-Q：由人工注释的公立学校数学问题，缺少一个变量赋值，（4）GSME-Q：GSM-Q的一个版本，其中文字问题由人工注释员翻译成方程。LLM的任务是从一组选项中选择正确的澄清问题。虽然最先进的模型在GSM-Q和GSME-Q上表现出色，但在Logic-Q和Planning-Q上的准确率仅为40-50%。分析表明，解决明确定义的推理问题的能力可能并不足以在我们的基准上取得成功：模型在确定应该提出什么问题方面存在困难，即使它们能够解决完全定义的版本的问题。此外，在Planning-Q领域，LLM倾向于不保留不确定性，即使明确提供了预测“不确定”的选项。这突显了对模型信息获取能力进行更深入研究的必要性。 

---
# ActionStudio: A Lightweight Framework for Data and Training of Action Models 

**Title (ZH)**: ActionStudio: 一种轻量级的动作模型数据与训练框架 

**Authors**: Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, Shiyu Wang, Tulika Awalgaonkar, Akshara Prabhakar, Haolin Chen, Weiran Yao, Zhiwei Liu, Juntao Tan, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22673)  

**Abstract**: Action models are essential for enabling autonomous agents to perform complex tasks. However, training large action models remains challenging due to the diversity of agent environments and the complexity of agentic data. Despite growing interest, existing infrastructure provides limited support for scalable, agent-specific fine-tuning. We present ActionStudio, a lightweight and extensible data and training framework designed for action models. ActionStudio unifies heterogeneous agent trajectories through a standardized format, supports diverse training paradigms including LoRA, full fine-tuning, and distributed setups, and integrates robust preprocessing and verification tools. We validate its effectiveness across both public and realistic industry benchmarks, demonstrating strong performance and practical scalability. We open-sourced code and data at this https URL to facilitate research in the community. 

**Abstract (ZH)**: 行动模型对于使自主代理能够执行复杂任务是必不可少的。然而，由于代理环境的多样性和代理数据的复杂性，训练大规模行动模型仍然具有挑战性。尽管现有兴趣不断增长，现有基础设施对可扩展的、针对代理特定的微调支持有限。我们提出了ActionStudio，一个轻量级且可扩展的数据和训练框架，专门设计用于行动模型。ActionStudio 通过标准化格式统一了异构代理轨迹，支持包括LoRA、全程微调和分布式设置在内的多种训练范式，并集成了稳健的预处理和验证工具。我们在公共和现实行业的基准测试中验证了其有效性，展示了强大的性能和实际的可扩展性。我们已在此<https://>URL 开放了代码和数据，以促进社区中的研究。 

---
# Unicorn: Text-Only Data Synthesis for Vision Language Model Training 

**Title (ZH)**: 独角兽：仅文本数据合成用于视觉语言模型训练 

**Authors**: Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22655)  

**Abstract**: Training vision-language models (VLMs) typically requires large-scale, high-quality image-text pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To tackle this, we propose a cross-integrated three-stage multimodal data synthesis framework, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse high-quality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multi-turn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLMs training. Code is available at this https URL. 

**Abstract (ZH)**: 训练视觉-语言模型（VLMs）通常需要大规模的高质量图像-文本配对数据，但收集或合成这样的数据成本高昂。相比之下，文本数据丰富且成本低廉，这促使我们问：是否可以纯粹从文本中合成高质量的多模态训练数据？为解决这一问题，我们提出了一种跨模态三阶段数据合成框架，生成两个数据集：Unicorn-1.2M和Unicorn-471K-Instruction。在第一阶段：多元化标题数据合成中，我们使用大型语言模型（LLMs）扩展稀疏的标题种子，构造出1.2M个语义多样的高质量标题。在第二阶段：指令调优数据生成中，我们将471K个标题进一步处理成多轮指令调优任务，以支持复杂的推理。最后，在第三阶段：模态表示迁移中，这些文本标题表示被转换为视觉表示，生成多样的合成图像表示。这一三阶段过程使我们能够在不依赖真实图像的情况下构建Unicorn-1.2M用于预训练和Unicorn-471K-Instruction用于指令调优。通过消除对真实图像的依赖性，同时保持数据质量和多样性，我们的框架为VLMs的训练提供了成本效益高且可扩展的解决方案。代码详见this https URL。 

---
# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models 

**Title (ZH)**: CPPO：加速基于群组相对策略优化的推理模型训练 

**Authors**: Zhihang Lin, Mingbao Lin, Yuan Xie, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.22342)  

**Abstract**: This paper introduces Completion Pruning Policy Optimization (CPPO) to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need for sampling multiple completions for each question. Our experiment and theoretical analysis reveals that the number of completions impacts model accuracy yet increases training time multiplicatively, and not all completions contribute equally to policy training -- their contribution depends on their relative advantage. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experimental results demonstrate that CPPO achieves up to $8.32\times$ speedup on GSM8K and $3.51\times$ on Math while preserving or even enhancing the accuracy compared to the original GRPO. We release our code at this https URL. 

**Abstract (ZH)**: 基于组相对策略优化的完成剪枝策略优化（CPPO）以加速推理模型训练 

---
# Agent-Centric Personalized Multiple Clustering with Multi-Modal LLMs 

**Title (ZH)**: 基于代理的多模态大型语言模型个性化多聚类 

**Authors**: Ziye Chen, Yiqun Duan, Riheng Zhu, Zhenbang Sun, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22241)  

**Abstract**: Personalized multiple clustering aims to generate diverse partitions of a dataset based on different user-specific aspects, rather than a single clustering. It has recently drawn research interest for accommodating varying user preferences. Recent approaches primarily use CLIP embeddings with proxy learning to extract representations biased toward user clustering preferences. However, CLIP primarily focuses on coarse image-text alignment, lacking a deep contextual understanding of user interests. To overcome these limitations, we propose an agent-centric personalized clustering framework that leverages multi-modal large language models (MLLMs) as agents to comprehensively traverse a relational graph to search for clusters based on user interests. Due to the advanced reasoning mechanism of MLLMs, the obtained clusters align more closely with user-defined criteria than those obtained from CLIP-based representations. To reduce computational overhead, we shorten the agents' traversal path by constructing a relational graph using user-interest-biased embeddings extracted by MLLMs. A large number of weakly connected edges can be filtered out based on embedding similarity, facilitating an efficient traversal search for agents. Experimental results show that the proposed method achieves NMI scores of 0.9667 and 0.9481 on the Card Order and Card Suits benchmarks, respectively, largely improving the SOTA model by over 140%. 

**Abstract (ZH)**: 个性化多聚类旨在根据不同的用户特定方面生成数据集的多样化分区，而非单一聚类。近年来，由于能够容纳不同的用户偏好，该领域引起了研究兴趣。现有的方法主要利用CLIP嵌入和代理学习来提取偏向用户聚类偏好的表示。然而，CLIP 主要关注粗粒度的图像-文本对齐，缺乏对用户兴趣的深入上下文理解。为克服这些局限性，我们提出了一种基于代理的个性化聚类框架，利用多模态大规模语言模型（MLLMs）作为代理，全面遍历关系图以根据用户兴趣搜索聚类。得益于MLLMs的高级推理机制，所获得的聚类与用户定义的标准更为一致，优于基于CLIP的表示。为了减少计算开销，我们通过使用MLLMs提取的兴趣偏向嵌入构建关系图，缩短代理的遍历路径。基于嵌入相似性可以过滤掉大量的弱连接边，便于代理的高效遍历搜索。实验结果表明，所提出的方法在Card Order和Card Suits基准上分别获得了0.9667和0.9481的NMI分数，显著优于当前最先进模型超过140%。 

---
# Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF 

**Title (ZH)**: Sharpe比率引导的主动学习方法在RLHF中的偏好优化 

**Authors**: Syrine Belakaria, Joshua Kazdan, Charles Marx, Chris Cundy, Willie Neiswanger, Sanmi Koyejo, Barbara E. Engelhardt, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2503.22137)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become a cornerstone of the training and alignment pipeline for large language models (LLMs). Recent advances, such as direct preference optimization (DPO), have simplified the preference learning step. However, collecting preference data remains a challenging and costly process, often requiring expert annotation. This cost can be mitigated by carefully selecting the data points presented for annotation. In this work, we propose an active learning approach to efficiently select prompt and preference pairs using a risk assessment strategy based on the Sharpe Ratio. To address the challenge of unknown preferences prior to annotation, our method evaluates the gradients of all potential preference annotations to assess their impact on model updates. These gradient-based evaluations enable risk assessment of data points regardless of the annotation outcome. By leveraging the DPO loss derivations, we derive a closed-form expression for computing these Sharpe ratios on a per-tuple basis, ensuring our approach remains both tractable and computationally efficient. We also introduce two variants of our method, each making different assumptions about prior information. Experimental results demonstrate that our method outperforms the baseline by up to 5% in win rates against the chosen completion with limited human preference data across several language models and real-world datasets. 

**Abstract (ZH)**: 从人类反馈中学习的强化学习（RLHF）已成为大型语言模型（LLMs）训练和对齐管道的基石。近期进展，如直接偏好优化（DPO），简化了偏好学习步。然而，偏好数据的收集仍然是一个具有挑战性和成本高昂的过程，往往需要专家标注。通过仔细选择用于标注的数据点，可以减轻这一成本。在这种工作中，我们提出了一种主动学习方法，以风险评估策略为基础（基于夏普比率）来高效选择提示和偏好对。为了解决标注前未知偏好的挑战，我们的方法评估了所有潜在偏好标注的梯度，评估其对模型更新的影响。基于梯度的评估使我们能够在不影响标注结果的情况下，对数据点进行风险评估。通过利用DPO损失的推导，我们推导出一个闭形式表达式来计算每个元组上的夏普比率，确保我们的方法既可操作性强又计算效率高。我们还引入了两种我们方法的变种，每种变种对先验信息做了不同的假设。实验结果表明，我们的方法在有限的人类偏好数据下，与选择的完成相比，胜出率最高可达5%，覆盖多个语言模型和真实世界数据集。 

---
# Multi-Task Semantic Communications via Large Models 

**Title (ZH)**: 大规模模型驱动的多任务语义通信 

**Authors**: Wanli Ni, Zhijin Qin, Haofeng Sun, Xiaoming Tao, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.22064)  

**Abstract**: Artificial intelligence (AI) promises to revolutionize the design, optimization and management of next-generation communication systems. In this article, we explore the integration of large AI models (LAMs) into semantic communications (SemCom) by leveraging their multi-modal data processing and generation capabilities. Although LAMs bring unprecedented abilities to extract semantics from raw data, this integration entails multifaceted challenges including high resource demands, model complexity, and the need for adaptability across diverse modalities and tasks. To overcome these challenges, we propose a LAM-based multi-task SemCom (MTSC) architecture, which includes an adaptive model compression strategy and a federated split fine-tuning approach to facilitate the efficient deployment of LAM-based semantic models in resource-limited networks. Furthermore, a retrieval-augmented generation scheme is implemented to synthesize the most recent local and global knowledge bases to enhance the accuracy of semantic extraction and content generation, thereby improving the inference performance. Finally, simulation results demonstrate the efficacy of the proposed LAM-based MTSC architecture, highlighting the performance enhancements across various downstream tasks under varying channel conditions. 

**Abstract (ZH)**: 人工智能（AI）有望革命化下一代通信系统的设计、优化和管理。本文探讨了通过利用其多模态数据处理和生成能力，将大规模人工智能模型（LAMs）集成到语义通信（SemCom）中的方法。尽管LAMs能够前所未有的从原始数据中提取语义，但这种集成带来了包括高资源需求、模型复杂性以及跨多种模态和任务的适应性在内的多方面挑战。为克服这些挑战，我们提出了一种基于LAM的多任务语义通信（MTSC）架构，该架构包括自适应模型压缩策略和联邦分裂微调方法，以促进在资源受限网络中高效部署基于LAM的语义模型。此外，我们实现了一种检索增强生成方案，综合最新的局部和全局知识库，以提高语义提取和内容生成的准确性，从而改善推理性能。最后，仿真结果证明了所提出的基于LAM的MTSC架构的有效性，在不同信道条件下，突显了其在各种下游任务中的性能提升。 

---
# OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment 

**Title (ZH)**: OntoAligner：一个全面的模块化和稳健的Python工具包用于本体对齐 

**Authors**: Hamed Babaei Giglou, Jennifer D'Souza, Oliver Karras, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2503.21902)  

**Abstract**: Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. We present OntoAligner, a comprehensive, modular, and robust Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. This paper details the design principles, architecture, and implementation of the OntoAligner, demonstrating its utility through benchmarks on standard OA tasks. Our evaluation highlights OntoAligner's ability to handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications. 

**Abstract (ZH)**: 本体对齐（OA）是实现跨异构知识系统语义互操作性的基础。我们提出OntoAligner，这是一个全面、模块化且稳健的Python工具包，旨在解决现有工具在实践中面临的局限性。现有工具在可扩展性、模块化以及与_recent AI进展_的集成方面存在局限。OntoAligner提供了一个灵活的架构，整合了现有的轻量级本体对齐技术，如模糊匹配，但更进一步支持了与检索增强生成及大型语言模型相关的当代方法。该框架注重可扩展性，让研究人员能够集成自定义对齐算法和数据集。本文详细介绍了OntoAligner的设计原则、架构和实现，并通过标准本体对齐任务的基准测试展示了其实用性。我们的评估突显了OntoAligner能够通过少量代码高效处理大规模本体，同时保持高质量对齐的能力。通过将OntoAligner开源，我们旨在为本体对齐社区提供一个促进创新和协作的资源，使研究人员和从业者能够利用该工具包开展可再现的本体对齐研究和实际应用。 

---
# Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment 

**Title (ZH)**: Best-of-N是否最优？推理时序对齐的覆盖面、扩展性和最优性探究 

**Authors**: Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Dylan J. Foster, Akshay Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21878)  

**Abstract**: Inference-time computation provides an important axis for scaling language model performance, but naively scaling compute through techniques like Best-of-$N$ sampling can cause performance to degrade due to reward hacking. Toward a theoretical understanding of how to best leverage additional computation, we focus on inference-time alignment which we formalize as the problem of improving a pre-trained policy's responses for a prompt of interest, given access to an imperfect reward model. We analyze the performance of inference-time alignment algorithms in terms of (i) response quality, and (ii) compute, and provide new results that highlight the importance of the pre-trained policy's coverage over high-quality responses for performance and compute scaling:
1. We show that Best-of-$N$ alignment with an ideal choice for $N$ can achieve optimal performance under stringent notions of coverage, but provably suffers from reward hacking when $N$ is large, and fails to achieve tight guarantees under more realistic coverage conditions.
2. We introduce $\texttt{InferenceTimePessimism}$, a new algorithm which mitigates reward hacking through deliberate use of inference-time compute, implementing the principle of pessimism in the face of uncertainty via rejection sampling; we prove that its performance is optimal and does not degrade with $N$, meaning it is scaling-monotonic.
We complement our theoretical results with an experimental evaluation that demonstrate the benefits of $\texttt{InferenceTimePessimism}$ across a variety of tasks and models. 

**Abstract (ZH)**: 推理时计算对于扩展语言模型性能提供了重要维度，但通过如Best-of-$N$采样等技术简单扩展计算会因奖励劫持导致性能下降。为理解和最佳利用额外计算，我们聚焦于推理时对齐问题，将其形式化为改进预训练策略对特定提示响应的问题，前提是可以访问一个不完美的奖励模型。我们从响应质量和计算量两个方面分析推理时对齐算法的性能，并提供了新的结果，强调预训练策略在高质量响应上的覆盖对于性能和计算扩展的重要性：

1. 我们证明，在严格的覆盖定义下，Best-of-$N$对齐在理想的选择$N$时可以实现最优性能，但当$N$较大时，理论上会受到奖励劫持的影响，并在更现实的覆盖条件下无法实现紧要的保障。
2. 我们引入了$\texttt{InferenceTimePessimism}$算法，通过故意利用推理时的计算来减轻奖励劫持，通过拒绝采样实现不确定性面前的悲观原则；我们证明其性能最优且不会因$N$的增加而恶化，这意味着它是计算扩展单调的。
我们通过实验证明了$\texttt{InferenceTimePessimism}$在各种任务和模型上的优势。 

---
# DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness 

**Title (ZH)**: DSO：通过模拟反馈对齐3D生成器以实现物理合理性 

**Authors**: Ruining Li, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22677)  

**Abstract**: Most 3D object generators focus on aesthetic quality, often neglecting physical constraints necessary in applications. One such constraint is that the 3D object should be self-supporting, i.e., remains balanced under gravity. Prior approaches to generating stable 3D objects used differentiable physics simulators to optimize geometry at test-time, which is slow, unstable, and prone to local optima. Inspired by the literature on aligning generative models to external feedback, we propose Direct Simulation Optimization (DSO), a framework to use the feedback from a (non-differentiable) simulator to increase the likelihood that the 3D generator outputs stable 3D objects directly. We construct a dataset of 3D objects labeled with a stability score obtained from the physics simulator. We can then fine-tune the 3D generator using the stability score as the alignment metric, via direct preference optimization (DPO) or direct reward optimization (DRO), a novel objective, which we introduce, to align diffusion models without requiring pairwise preferences. Our experiments show that the fine-tuned feed-forward generator, using either DPO or DRO objective, is much faster and more likely to produce stable objects than test-time optimization. Notably, the DSO framework works even without any ground-truth 3D objects for training, allowing the 3D generator to self-improve by automatically collecting simulation feedback on its own outputs. 

**Abstract (ZH)**: Most 3D对象生成器专注于美学质量，往往忽视了应用中必要的物理约束。其中之一是3D对象应具备自支撑性，即在重力作用下保持平衡。先前用于生成稳定3D对象的方法使用可微物理模拟器在测试时优化几何形状，这导致速度慢、不稳定并且容易陷入局部最优。受到将生成模型与外部反馈对齐文献的启发，我们提出了一种直接模拟优化（DSO）框架，该框架利用非可微模拟器的反馈直接增加3D生成器输出稳定3D对象的可能性。我们构建了一个带有稳定分数标签的3D对象数据集，该稳定分数由物理模拟器获得。然后，我们可以通过直接偏好优化（DPO）或直接奖励优化（DRO）——这是一种我们新提出的不需要成对偏好即可对扩散模型进行对齐的方法——使用稳定分数作为对齐度量来微调3D生成器。我们的实验表明，使用DPO或DRO目标函数的微调前馈生成器比在测试时优化生成更快速且更有可能产生稳定对象。值得注意的是，DSO框架甚至在没有真实3D对象用于训练的情况下也能有效工作，允许3D生成器通过自动收集自身输出的模拟反馈来自我改进。 

---
# Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation 

**Title (ZH)**: 深思熟虑再推荐：激发顺序推荐中的潜在推理能力 

**Authors**: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22675)  

**Abstract**: Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 

**Abstract (ZH)**: 序贯推荐（SeqRec）旨在通过捕捉用户历史交互中的序贯模式来预测下一个项目，在许多实际推荐系统中发挥着重要作用。然而，现有的方法主要采用了直接前向计算范式，其中序列编码的最终隐藏状态作为用户表示。我们 argue 认为，这种推理范式由于计算深度有限，难以模拟用户偏好的复杂演变性质，并且缺乏对长尾项目的精细理解，导致性能不佳。为解决这一问题，我们提出了 ReaRec，这是第一个推荐系统的推理时计算框架，通过隐式的多步推理增强用户表示。具体而言，ReaRec 自回归地将序列的最后一个隐藏状态输入到序贯推荐器中，并结合特殊的推理位置嵌入以解耦原始项目编码空间与多步推理空间。此外，我们引入了两种轻量级的基于推理的学习方法，即集成推理学习（ERL）和逐步推理学习（PRL），以进一步有效地充分发挥 ReaRec 的推理潜力。在五个公开的实际数据集和不同 SeqRec 架构上的 extensive 实验表明，我们提出的 ReaRec 具有通用性和有效性。值得注意的是，事后分析显示，ReaRec 显著提高了多个序贯推荐基础架构的性能天花板，大约为 30%-50%。因此，我们相信这项工作可以开启未来推理时计算在序贯推荐中研究的新途径。 

---
# Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers 

**Title (ZH)**: 探索多阶段微调对交叉编码重新排ranking器有效性的影响 

**Authors**: Francesca Pezzuti, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22672)  

**Abstract**: State-of-the-art cross-encoders can be fine-tuned to be highly effective in passage re-ranking. The typical fine-tuning process of cross-encoders as re-rankers requires large amounts of manually labelled data, a contrastive learning objective, and a set of heuristically sampled negatives. An alternative recent approach for fine-tuning instead involves teaching the model to mimic the rankings of a highly effective large language model using a distillation objective. These fine-tuning strategies can be applied either individually, or in sequence. In this work, we systematically investigate the effectiveness of point-wise cross-encoders when fine-tuned independently in a single stage, or sequentially in two stages. Our experiments show that the effectiveness of point-wise cross-encoders fine-tuned using contrastive learning is indeed on par with that of models fine-tuned with multi-stage approaches. Code is available for reproduction at this https URL. 

**Abstract (ZH)**: 最先进的跨编码器可以调整为在段落排序中非常有效的细调过程。跨编码器作为重新排序器的典型细调过程需要大量手动标注数据、对比学习目标以及一组启发式采样的负样本。一种最近的替代细调方法是通过蒸馏目标让模型模仿高性能大型语言模型的排序。这些细调策略可以单独应用，也可以按顺序分两个阶段应用。在本工作中，我们系统研究了单阶段独立细调以及两阶段顺序细调条件下点-wise跨编码器的有效性。实验结果显示，使用对比学习方式细调的点-wise跨编码器的效果确实与多阶段细调方法相当。相关代码可在以下网址复制：this https URL。 

---
# Evaluation of Machine-generated Biomedical Images via A Tally-based Similarity Measure 

**Title (ZH)**: 基于计数的相似度量评价机器生成的生物医学图像 

**Authors**: Frank J. Brooks, Rucha Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2503.22658)  

**Abstract**: Super-resolution, in-painting, whole-image generation, unpaired style-transfer, and network-constrained image reconstruction each include an aspect of machine-learned image synthesis where the actual ground truth is not known at time of use. It is generally difficult to quantitatively and authoritatively evaluate the quality of synthetic images; however, in mission-critical biomedical scenarios robust evaluation is paramount. In this work, all practical image-to-image comparisons really are relative qualifications, not absolute difference quantifications; and, therefore, meaningful evaluation of generated image quality can be accomplished using the Tversky Index, which is a well-established measure for assessing perceptual similarity. This evaluation procedure is developed and then demonstrated using multiple image data sets, both real and simulated. The main result is that when the subjectivity and intrinsic deficiencies of any feature-encoding choice are put upfront, Tversky's method leads to intuitive results, whereas traditional methods based on summarizing distances in deep feature spaces do not. 

**Abstract (ZH)**: 超分辨率、 inpainting、整图生成、无配对风格转移以及网络约束图像重建各自包含机器学习生成图像的一个方面，其中实际的ground truth在使用时未知。通常难以定量和权威性地评估合成图像的质量；然而，在关键的生物医学场景中，稳健的评估至关重要。在本工作中，所有实际的图像到图像比较实际上是相对的评价，而不是绝对差异的量化；因此，可以使用Tversky指数来评估生成图像质量，该指数是一个已建立的感知相似性评估措施。该评估程序通过多种实际和模拟图像数据集得到开发和验证。主要结果是，当将任何特征编码选择的主观性和内在缺陷摆到台面上时，Tversky的方法会产生直观的结果，而基于深入特征空间距离汇总的传统方法则不会。 

---
# Empirical Analysis of Sim-and-Real Cotraining Of Diffusion Policies For Planar Pushing from Pixels 

**Title (ZH)**: 基于像素的平面推物任务中模拟与真实联合训练扩散策略的实证分析 

**Authors**: Adam Wei, Abhinav Agarwal, Boyuan Chen, Rohan Bosworth, Nicholas Pfaff, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.22634)  

**Abstract**: In imitation learning for robotics, cotraining with demonstration data generated both in simulation and on real hardware has emerged as a powerful recipe to overcome the sim2real gap. This work seeks to elucidate basic principles of this sim-and-real cotraining to help inform simulation design, sim-and-real dataset creation, and policy training. Focusing narrowly on the canonical task of planar pushing from camera inputs enabled us to be thorough in our study. These experiments confirm that cotraining with simulated data \emph{can} dramatically improve performance in real, especially when real data is limited. Performance gains scale with simulated data, but eventually plateau; real-world data increases this performance ceiling. The results also suggest that reducing the domain gap in physics may be more important than visual fidelity for non-prehensile manipulation tasks. Perhaps surprisingly, having some visual domain gap actually helps the cotrained policy -- binary probes reveal that high-performing policies learn to distinguish simulated domains from real. We conclude by investigating this nuance and mechanisms that facilitate positive transfer between sim-and-real. In total, our experiments span over 40 real-world policies (evaluated on 800+ trials) and 200 simulated policies (evaluated on 40,000+ trials). 

**Abstract (ZH)**: 在机器人学中通过模拟与真实硬件数据协同训练以克服模拟到现实的差距：探究基本原理及其对仿真设计、仿真与现实数据集创建及策略训练的指导意义。 

---
# Challenges and Paths Towards AI for Software Engineering 

**Title (ZH)**: 面向软件工程的AI挑战与途径 

**Authors**: Alex Gu, Naman Jain, Wen-Ding Li, Manish Shetty, Yijia Shao, Ziyang Li, Diyi Yang, Kevin Ellis, Koushik Sen, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2503.22625)  

**Abstract**: AI for software engineering has made remarkable progress recently, becoming a notable success within generative AI. Despite this, there are still many challenges that need to be addressed before automated software engineering reaches its full potential. It should be possible to reach high levels of automation where humans can focus on the critical decisions of what to build and how to balance difficult tradeoffs while most routine development effort is automated away. Reaching this level of automation will require substantial research and engineering efforts across academia and industry. In this paper, we aim to discuss progress towards this in a threefold manner. First, we provide a structured taxonomy of concrete tasks in AI for software engineering, emphasizing the many other tasks in software engineering beyond code generation and completion. Second, we outline several key bottlenecks that limit current approaches. Finally, we provide an opinionated list of promising research directions toward making progress on these bottlenecks, hoping to inspire future research in this rapidly maturing field. 

**Abstract (ZH)**: AI在软件工程中的应用取得了显著进展，成为生成型AI的显著成功案例。尽管如此，在实现完全自动化的软件工程之前，仍有许多挑战需要解决。应有可能实现高度自动化，使人类专注于构建的重要决策和复杂的权衡取舍，而大多数常规开发工作被自动化取代。达到这一水平的自动化将需要学术界和工业界的大量研究和工程努力。在本文中，我们旨在以三方面的方式探讨这一目标的进展。首先，我们提供了一种结构化的AI在软件工程中具体任务的分类法，强调软件工程中除了代码生成和完成之外的许多其他任务。其次，我们概述了当前方法的几个关键瓶颈。最后，我们提出了一种有倾向性的研究方向列表，旨在为解决这些瓶颈提供灵感，希望激发这一快速成熟领域的未来研究。 

---
# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users 

**Title (ZH)**: 评估多模态语言模型作为视觉辅助工具用于视觉障碍用户 

**Authors**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22610)  

**Abstract**: This paper explores the effectiveness of Multimodal Large Language models (MLLMs) as assistive technologies for visually impaired individuals. We conduct a user survey to identify adoption patterns and key challenges users face with such technologies. Despite a high adoption rate of these models, our findings highlight concerns related to contextual understanding, cultural sensitivity, and complex scene understanding, particularly for individuals who may rely solely on them for visual interpretation. Informed by these results, we collate five user-centred tasks with image and video inputs, including a novel task on Optical Braille Recognition. Our systematic evaluation of twelve MLLMs reveals that further advancements are necessary to overcome limitations related to cultural context, multilingual support, Braille reading comprehension, assistive object recognition, and hallucinations. This work provides critical insights into the future direction of multimodal AI for accessibility, underscoring the need for more inclusive, robust, and trustworthy visual assistance technologies. 

**Abstract (ZH)**: 本文探讨了多模态大型语言模型（MLLMs）作为盲人辅助技术的有效性。我们开展了一项用户调查，以识别用户采用模式和他们使用此类技术所面临的key挑战。尽管这些模型的采用率很高，但我们的研究结果突显了与语境理解、文化敏感性和复杂场景理解相关的问题，尤其是对于那些可能完全依赖它们进行视觉解释的个人。根据这些结果，我们汇总了五个以图像和视频为输入的用户中心任务，其中包括一项新颖的光学盲文识别任务。对十二种MLLMs的系统性评估表明，为了克服与文化背景、多语言支持、盲文阅读理解、辅助对象识别和幻觉相关的问题，还需要进一步的发展。本研究为多模态AI在无障碍领域的未来方向提供了关键见解，强调了需要更多包容性、稳健性和可信度的视觉辅助技术。 

---
# Generative Latent Neural PDE Solver using Flow Matching 

**Title (ZH)**: 生成型潜神经PDE求解器基于流匹配 

**Authors**: Zijie Li, Anthony Zhou, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2503.22600)  

**Abstract**: Autoregressive next-step prediction models have become the de-facto standard for building data-driven neural solvers to forecast time-dependent partial differential equations (PDEs). Denoise training that is closely related to diffusion probabilistic model has been shown to enhance the temporal stability of neural solvers, while its stochastic inference mechanism enables ensemble predictions and uncertainty quantification. In principle, such training involves sampling a series of discretized diffusion timesteps during both training and inference, inevitably increasing computational overhead. In addition, most diffusion models apply isotropic Gaussian noise on structured, uniform grids, limiting their adaptability to irregular domains. We propose a latent diffusion model for PDE simulation that embeds the PDE state in a lower-dimensional latent space, which significantly reduces computational costs. Our framework uses an autoencoder to map different types of meshes onto a unified structured latent grid, capturing complex geometries. By analyzing common diffusion paths, we propose to use a coarsely sampled noise schedule from flow matching for both training and testing. Numerical experiments show that the proposed model outperforms several deterministic baselines in both accuracy and long-term stability, highlighting the potential of diffusion-based approaches for robust data-driven PDE learning. 

**Abstract (ZH)**: 自回归下一步预测模型已成为构建数据驱动神经求解器以预测时间依赖偏微分方程(PDEs)的标准方法。与去噪训练紧密相关的去噪训练概率模型已被证明能够增强神经求解器的时间稳定性，其随机推断机制还能够实现 Ensemble 预测和不确定性量化。原则上，这种训练在训练和推理过程中不可避免地需要采样一系列离散化的扩散时间步长，从而增加计算开销。此外，大多数扩散模型在规则的均匀网格上应用各向同性高斯噪声，限制了它们对不规则领域域的适应性。我们提出了一种用于PDE模拟的潜在扩散模型，将PDE状态嵌入到低维潜在空间中，这显著降低了计算成本。该框架使用自编码器将不同类型的网格映射到统一的结构化潜在网格上，捕获复杂几何结构。通过分析常见的扩散路径，我们提出在训练和测试中使用从流动匹配粗采样的噪声调度。数值实验表明，所提出模型在准确性和长期稳定性方面均优于几种确定性基线，突显了基于扩散的方法在鲁棒数据驱动PDE学习中的潜力。 

---
# KEVS: Enhancing Segmentation of Visceral Adipose Tissue in Pre-Cystectomy CT with Gaussian Kernel Density Estimation 

**Title (ZH)**: KEVS: 用高斯核密度估计增强肾切除术CT前期内脏脂肪组织分割 

**Authors**: Thomas Boucher, Nicholas Tetlow, Annie Fung, Amy Dewar, Pietro Arina, Sven Kerneis, John Whittle, Evangelos B. Mazomenos  

**Link**: [PDF](https://arxiv.org/pdf/2503.22592)  

**Abstract**: Purpose: The distribution of visceral adipose tissue (VAT) in cystectomy patients is indicative of the incidence of post-operative complications. Existing VAT segmentation methods for computed tomography (CT) employing intensity thresholding have limitations relating to inter-observer variability. Moreover, the difficulty in creating ground-truth masks limits the development of deep learning (DL) models for this task. This paper introduces a novel method for VAT prediction in pre-cystectomy CT, which is fully automated and does not require ground-truth VAT masks for training, overcoming aforementioned limitations. Methods: We introduce the Kernel density Enhanced VAT Segmentator ( KEVS), combining a DL semantic segmentation model, for multi-body feature prediction, with Gaussian kernel density estimation analysis of predicted subcutaneous adipose tissue to achieve accurate scan-specific predictions of VAT in the abdominal cavity. Uniquely for a DL pipeline, KEVS does not require ground-truth VAT masks. Results: We verify the ability of KEVS to accurately segment abdominal organs in unseen CT data and compare KEVS VAT segmentation predictions to existing state-of-the-art (SOTA) approaches in a dataset of 20 pre-cystectomy CT scans, collected from University College London Hospital (UCLH-Cyst), with expert ground-truth annotations. KEVS presents a 4.80% and 6.02% improvement in Dice Coefficient over the second best DL and thresholding-based VAT segmentation techniques respectively when evaluated on UCLH-Cyst. Conclusion: This research introduces KEVS; an automated, SOTA method for the prediction of VAT in pre-cystectomy CT which eliminates inter-observer variability and is trained entirely on open-source CT datasets which do not contain ground-truth VAT masks. 

**Abstract (ZH)**: 目的：膀胱切除术患者腹内脂肪组织（VAT）的分布是术后并发症发生的指示器。现有的基于CT的VAT分割方法使用强度阈值分割存在因观察者间差异而导致的局限性。此外，创建ground-truth掩模的难度限制了该任务深度学习（DL）模型的发展。本文提出了一种新的用于膀胱切除术前CT中VAT预测的方法，该方法完全自动化且无需训练时使用ground-truth VAT掩模，克服了上述局限性。方法：我们引入了一种名为Kernel density Enhanced VAT Segmentator（KEVS）的方法，结合了一个基于DL的语义分割模型用于多体素特征预测，并通过高斯核密度估计分析预测的皮下脂肪组织以在腹腔中实现准确的扫描特定VAT分割。唯一不同的是，KEVS无需ground-truth VAT掩模。结果：我们验证了KEVS在未见过的CT数据中准确分割腹部器官的能力，并将KEVS的VAT分割预测与20例来自University College London Hospital（UCLH-Cyst）的膀胱切除术前CT扫描数据集中的现有最佳方法进行了比较，该数据集具有专家ground-truth注释。当在UCLH-Cyst数据集上评估时，KEVS分别在第二好的DL方法和基于阈值的VAT分割技术上分别取得了4.80%和6.02%的Dice系数改进。结论：本文介绍了一种自动化的、基于开源CT数据集训练、无需ground-truth VAT掩模的最新方法KEVS，用于膀胱切除术前CT中VAT的预测，消除了观察者间差异。 

---
# Using AI to Summarize US Presidential Campaign TV Advertisement Videos, 1952-2012 

**Title (ZH)**: 使用AI总结1952-2012年美国总统竞选电视广告视频摘要 

**Authors**: Adam Breuer, Bryce J. Dietrich, Michael H. Crespin, Matthew Butler, J.A. Pyrse, Kosuke Imai  

**Link**: [PDF](https://arxiv.org/pdf/2503.22589)  

**Abstract**: This paper introduces the largest and most comprehensive dataset of US presidential campaign television advertisements, available in digital format. The dataset also includes machine-searchable transcripts and high-quality summaries designed to facilitate a variety of academic research. To date, there has been great interest in collecting and analyzing US presidential campaign advertisements, but the need for manual procurement and annotation led many to rely on smaller subsets. We design a large-scale parallelized, AI-based analysis pipeline that automates the laborious process of preparing, transcribing, and summarizing videos. We then apply this methodology to the 9,707 presidential ads from the Julian P. Kanter Political Commercial Archive. We conduct extensive human evaluations to show that these transcripts and summaries match the quality of manually generated alternatives. We illustrate the value of this data by including an application that tracks the genesis and evolution of current focal issue areas over seven decades of presidential elections. Our analysis pipeline and codebase also show how to use LLM-based tools to obtain high-quality summaries for other video datasets. 

**Abstract (ZH)**: 本文介绍了迄今为止最大和最全面的美国总统竞选电视广告数据集，数据集以数字格式提供。该数据集还包括机器可搜索的脚本和高质量的摘要，旨在促进各种学术研究。迄今为止，收集和分析美国总统竞选广告引起了极大的兴趣，但由于需要人工采购和标注，许多研究者依赖于较小的数据子集。我们设计了一种大规模并行化的基于AI的分析管道，自动完成了准备、转录和摘要视频的繁琐过程。然后，我们将这种方法应用于朱利安·P·卡纳特政治商业档案中的9,707条总统广告。我们进行了广泛的人机评估，以证明这些转录和摘要与手工生成的替代品具有相同的质量水平。我们通过一个示例应用展示了这些数据的价值，该应用追踪了七十年来总统选举中当前焦点议题的起源和发展。我们的分析管道和代码库还展示了如何使用基于LLM的工具为其他视频数据集获得高质量的摘要。 

---
# Historical Ink: Exploring Large Language Models for Irony Detection in 19th-Century Spanish 

**Title (ZH)**: 历史墨迹：探索大型语言模型在19世纪西班牙语讽刺检测中的应用 

**Authors**: Kevin Cohen, Laura Manrique-Gómez, Rubén Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2503.22585)  

**Abstract**: This study explores the use of large language models (LLMs) to enhance datasets and improve irony detection in 19th-century Latin American newspapers. Two strategies were employed to evaluate the efficacy of BERT and GPT-4o models in capturing the subtle nuances nature of irony, through both multi-class and binary classification tasks. First, we implemented dataset enhancements focused on enriching emotional and contextual cues; however, these showed limited impact on historical language analysis. The second strategy, a semi-automated annotation process, effectively addressed class imbalance and augmented the dataset with high-quality annotations. Despite the challenges posed by the complexity of irony, this work contributes to the advancement of sentiment analysis through two key contributions: introducing a new historical Spanish dataset tagged for sentiment analysis and irony detection, and proposing a semi-automated annotation methodology where human expertise is crucial for refining LLMs results, enriched by incorporating historical and cultural contexts as core features. 

**Abstract (ZH)**: 本研究探讨了使用大规模语言模型（LLMs）以增强数据集并提高对19世纪拉丁美洲报纸中讽刺的检测能力。通过多类和二分类任务评估了BERT和GPT-4o模型在捕捉讽刺微妙性质方面的有效性，采用了两种策略。首先，我们实施了专注于丰富情感和上下文线索的数据集增强，但这些方法在历史语言分析方面的效果有限。其次，我们采用了一种半自动注释过程，有效地解决了类别不平衡问题，并通过高质量的注释丰富了数据集。尽管讽刺的复杂性带来了挑战，本研究仍通过两个关键贡献推进了情感分析：一是引入了一个新的情感标注历史西班牙语数据集，用于讽刺检测；二是提出了结合历史和文化背景的半自动注释方法，这种方法依靠人类专业知识来细化LLMs的结果。 

---
# Breaking Language Barriers in Visual Language Models via Multilingual Textual Regularization 

**Title (ZH)**: 通过多语言文本正则化打破语言壁垒的视觉语言模型方法 

**Authors**: Iñigo Pikabea, Iñaki Lacunza, Oriol Pareras, Carlos Escolano, Aitor Gonzalez-Agirre, Javier Hernando, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2503.22577)  

**Abstract**: Rapid advancements in Visual Language Models (VLMs) have transformed multimodal understanding but are often constrained by generating English responses regardless of the input language. This phenomenon has been termed as Image-induced Fidelity Loss (IFL) and stems from limited multimodal multilingual training data. To address this, we propose a continuous multilingual integration strategy that injects text-only multilingual data during visual instruction tuning, preserving the language model's original multilingual capabilities. Extensive evaluations demonstrate that our approach significantly improves linguistic fidelity across languages without degradation in visual performance. We also explore model merging, which improves language fidelity but comes at the cost of visual performance. In contrast, our core method achieves robust multilingual alignment without trade-offs, offering a scalable and effective path to mitigating IFL for global VLM adoption. 

**Abstract (ZH)**: 快速发展的视觉语言模型(VLMs)已 transforming 多模态理解，但通常受限于无论输入语言为何种，生成英语响应的现象。这一现象被称为图像引发的忠实度损失(IFL)，源自于多模态多语言训练数据的限制。为解决此问题，我们提出了一种连续的多语言集成策略，在视觉指令调整期间注入仅文本的多语言数据，从而保留语言模型原有的多语言能力。广泛评估表明，我们的方法显著提升了多种语言的语义忠实度，同时不牺牲视觉性能。我们还探讨了模型合并，这可以提升语言忠实度，但会牺牲视觉性能。相比之下，我们的核心方法实现了稳健的多语言对齐，无需权衡，提供了一条可扩展且有效的减轻IFL的道路，促进全球VLM的采用。 

---
# On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations 

**Title (ZH)**: 关于可互换深度强化学习实现的错误假设 

**Authors**: Rajdeep Singh Hundal, Yan Xiao, Xiaochun Cao, Jin Song Dong, Manuel Rigger  

**Link**: [PDF](https://arxiv.org/pdf/2503.22575)  

**Abstract**: Deep Reinforcement Learning (DRL) is a paradigm of artificial intelligence where an agent uses a neural network to learn which actions to take in a given environment. DRL has recently gained traction from being able to solve complex environments like driving simulators, 3D robotic control, and multiplayer-online-battle-arena video games. Numerous implementations of the state-of-the-art algorithms responsible for training these agents, like the Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, currently exist. However, studies make the mistake of assuming implementations of the same algorithm to be consistent and thus, interchangeable. In this paper, through a differential testing lens, we present the results of studying the extent of implementation inconsistencies, their effect on the implementations' performance, as well as their impact on the conclusions of prior studies under the assumption of interchangeable implementations. The outcomes of our differential tests showed significant discrepancies between the tested algorithm implementations, indicating that they are not interchangeable. In particular, out of the five PPO implementations tested on 56 games, three implementations achieved superhuman performance for 50% of their total trials while the other two implementations only achieved superhuman performance for less than 15% of their total trials. As part of a meticulous manual analysis of the implementations' source code, we analyzed implementation discrepancies and determined that code-level inconsistencies primarily caused these discrepancies. Lastly, we replicated a study and showed that this assumption of implementation interchangeability was sufficient to flip experiment outcomes. Therefore, this calls for a shift in how implementations are being used. 

**Abstract (ZH)**: 深度强化学习（DRL）中实现不一致性的影响研究：通过差异测试考察算法实现差异及其后果 

---
# A Framework for Cryptographic Verifiability of End-to-End AI Pipelines 

**Title (ZH)**: 端到端人工智能管道的 cryptographic 可验证性框架 

**Authors**: Kar Balan, Robert Learney, Tim Wood  

**Link**: [PDF](https://arxiv.org/pdf/2503.22573)  

**Abstract**: The increasing integration of Artificial Intelligence across multiple industry sectors necessitates robust mechanisms for ensuring transparency, trust, and auditability of its development and deployment. This topic is particularly important in light of recent calls in various jurisdictions to introduce regulation and legislation on AI safety. In this paper, we propose a framework for complete verifiable AI pipelines, identifying key components and analyzing existing cryptographic approaches that contribute to verifiability across different stages of the AI lifecycle, from data sourcing to training, inference, and unlearning. This framework could be used to combat misinformation by providing cryptographic proofs alongside AI-generated assets to allow downstream verification of their provenance and correctness. Our findings underscore the importance of ongoing research to develop cryptographic tools that are not only efficient for isolated AI processes, but that are efficiently `linkable' across different processes within the AI pipeline, to support the development of end-to-end verifiable AI technologies. 

**Abstract (ZH)**: 跨多个行业领域的人工智能日益集成 necessitates  robust 机制以确保其开发和部署的透明性、可信度和可审计性。鉴于各司法管辖区最近对人工智能安全引入监管和立法的呼吁，该主题尤为重要。本文提出了一种完整的可验证人工智能管道框架，识别关键组件并分析贯穿人工智能生命周期各阶段的现有密码学方法，以提高不同阶段的可验证性，从数据来源到训练、推理和遗忘。该框架可通过提供与人工智能生成资产一同的密码学证明，以供下游验证其来源和正确性，从而对抗虚假信息。我们的研究结果强调了持续开发不仅适用于孤立人工智能进程的高效密码学工具的重要性，还强调了这些工具在人工智能管道内不同过程中高效“关联”的重要性，以支持端到端可验证人工智能技术的发展。 

---
# Niyama : Breaking the Silos of LLM Inference Serving 

**Title (ZH)**: Niyama: 打破LLM推理服务的孤岛效应 

**Authors**: Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Shreyas Anupindi, Ramachandran Ramjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.22562)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has enabled diverse applications with very different latency requirements. Existing LLM serving frameworks rely on siloed infrastructure with coarse-grained workload segregation -- interactive and batch -- leading to inefficient resource utilization and limited support for fine-grained Quality-of-Service (QoS) differentiation. This results in operational inefficiencies, over-provisioning and poor load management during traffic surges.
We present Niyama, a novel QoS-driven inference serving system that enables efficient co-scheduling of diverse workloads on shared infrastructure. Niyama introduces fine-grained QoS classification allowing applications to specify precise latency requirements, and dynamically adapts scheduling decisions based on real-time system state. Leveraging the predictable execution characteristics of LLM inference, Niyama implements a dynamic chunking mechanism to improve overall throughput while maintaining strict QoS guarantees. Additionally, Niyama employs a hybrid prioritization policy that balances fairness and efficiency, and employs selective request relegation that enables graceful service degradation during overload conditions. Our evaluation demonstrates that Niyama increases serving capacity by 32% compared to current siloed deployments, while maintaining QoS guarantees. Notably, under extreme load, our system reduces SLO violations by an order of magnitude compared to current strategies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用使得具有非常不同延迟要求的多种应用成为可能。现有的LLM服务框架依赖于孤立的基础设施和粗粒度的工作负载分割——交互式和批量处理，导致资源利用率低下且难以提供细粒度的服务质量（QoS）差异化。这导致了操作效率低下、过度配置和业务高峰期间的负载管理不佳。

我们提出了Niyama，这是一种新颖的QoS驱动的推理服务系统，能够在共享基础设施上高效地协同调度多样化的工作负载。Niyama引入了细粒度的QoS分类机制，允许应用程序明确规定精确的延迟要求，并根据实时系统状态动态调整调度决策。利用LLM推理可预测的执行特性，Niyama实现了一种动态切片机制，以提高总体吞吐量同时保持严格的QoS保证。此外，Niyama采用了混合优先级策略来平衡公平性和效率，并使用选择性的请求降级策略，在过载情况下提供优雅的服务降级。我们的评估结果表明，与当前的孤立部署相比，Niyama的服务容量增加了32%，同时维持了QoS保证。特别是在极端负载下，我们的系统将SLO违规降低了十倍，优于当前策略。 

---
# SafeCast: Risk-Responsive Motion Forecasting for Autonomous Vehicles 

**Title (ZH)**: SafeCast：响应风险的自主车辆运动预测 

**Authors**: Haicheng Liao, Hanlin Kong, Bin Rao, Bonan Wang, Chengyue Wang, Guyang Yu, Yuming Huang, Ruru Tang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22541)  

**Abstract**: Accurate motion forecasting is essential for the safety and reliability of autonomous driving (AD) systems. While existing methods have made significant progress, they often overlook explicit safety constraints and struggle to capture the complex interactions among traffic agents, environmental factors, and motion dynamics. To address these challenges, we present SafeCast, a risk-responsive motion forecasting model that integrates safety-aware decision-making with uncertainty-aware adaptability. SafeCast is the first to incorporate the Responsibility-Sensitive Safety (RSS) framework into motion forecasting, encoding interpretable safety rules--such as safe distances and collision avoidance--based on traffic norms and physical principles. To further enhance robustness, we introduce the Graph Uncertainty Feature (GUF), a graph-based module that injects learnable noise into Graph Attention Networks, capturing real-world uncertainties and enhancing generalization across diverse scenarios. We evaluate SafeCast on four real-world benchmark datasets--Next Generation Simulation (NGSIM), Highway Drone (HighD), ApolloScape, and the Macao Connected Autonomous Driving (MoCAD)--covering highway, urban, and mixed-autonomy traffic environments. Our model achieves state-of-the-art (SOTA) accuracy while maintaining a lightweight architecture and low inference latency, underscoring its potential for real-time deployment in safety-critical AD systems. 

**Abstract (ZH)**: 准确的运动预测对于自主驾驶（AD）系统的安全性和可靠性至关重要。现有方法虽取得了显著进展，但往往忽视了明确的安全约束，并难以捕捉交通代理、环境因素和运动动力学之间的复杂交互。为应对这些挑战，我们提出了SafeCast，一种响应风险的运动预测模型，将安全意识决策与不确定性意识适应性相结合。SafeCast是首次将责任敏感安全（RSS）框架应用于运动预测，基于交通规范和物理原理编码可解释的安全规则，如安全距离和碰撞避免。为了进一步增强鲁棒性，我们引入了基于图的不确定性特征（GUF），这是一种基于图的模块，向图注意网络中注入可学习的噪声，捕捉现实世界的不确定性，提升不同场景下的泛化能力。我们在Next Generation Simulation（NGSIM）、Highway Drone（HighD）、ApolloScape和Macao Connected Autonomous Driving（MoCAD）四个真实世界基准数据集上评估了SafeCast，涵盖了高速公路、城市和混合自主交通环境。我们的模型在保持轻量级架构和低推理延迟的同时达到了最先进的（SOTA）准确度，突显了其在安全关键型AD系统中实时部署的潜力。 

---
# LIM: Large Interpolator Model for Dynamic Reconstruction 

**Title (ZH)**: 大插值模型用于动态重建 

**Authors**: Remy Sabathier, Niloy J. Mitra, David Novotny  

**Link**: [PDF](https://arxiv.org/pdf/2503.22537)  

**Abstract**: Reconstructing dynamic assets from video data is central to many in computer vision and graphics tasks. Existing 4D reconstruction approaches are limited by category-specific models or slow optimization-based methods. Inspired by the recent Large Reconstruction Model (LRM), we present the Large Interpolation Model (LIM), a transformer-based feed-forward solution, guided by a novel causal consistency loss, for interpolating implicit 3D representations across time. Given implicit 3D representations at times $t_0$ and $t_1$, LIM produces a deformed shape at any continuous time $t\in[t_0,t_1]$, delivering high-quality interpolated frames in seconds. Furthermore, LIM allows explicit mesh tracking across time, producing a consistently uv-textured mesh sequence ready for integration into existing production pipelines. We also use LIM, in conjunction with a diffusion-based multiview generator, to produce dynamic 4D reconstructions from monocular videos. We evaluate LIM on various dynamic datasets, benchmarking against image-space interpolation methods (e.g., FiLM) and direct triplane linear interpolation, and demonstrate clear advantages. In summary, LIM is the first feed-forward model capable of high-speed tracked 4D asset reconstruction across diverse categories. 

**Abstract (ZH)**: 从视频数据中重构动态资产是计算机视觉和图形任务中的核心问题。现有的4D重构方法受限于类别特定模型或慢速的优化方法。受到最近提出的大型重构模型（LRM）的启发，我们提出了大型插值模型（LIM），这是一种基于变压器的前馈解决方案，并由一个新的因果一致性损失引导，用于在时间上插值隐式的3D表示。给定时间$t_0$和$t_1$的隐式3D表示，LIM可在任意连续时间$t\in[t_0,t_1]$生成变形的形状，并在秒内提供高质量的插值帧。此外，LIM允许在时间上的显式网格追踪，生成可用于现有生产流水线的连续uv纹理网格序列。我们也使用LIM结合基于扩散的多视图生成器，从单目视频中生成动态4D重构。我们对各种动态数据集评估了LIM，将其与图像空间插值方法（例如，FiLM）和直接三平面线性插值进行基准测试，并展示了明显的优越性。总结来说，LIM是首个能够高速跨类别进行跟踪的4D资产重构的前馈模型。 

---
# AnnoPage Dataset: Dataset of Non-Textual Elements in Documents with Fine-Grained Categorization 

**Title (ZH)**: AnnoPage 数据集：文档中非文本元素的细粒度分类数据集 

**Authors**: Martin Kišš, Michal Hradiš, Martina Dvořáková, Václav Jiroušek, Filip Kersch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22526)  

**Abstract**: We introduce the AnnoPage Dataset, a novel collection of 7550 pages from historical documents, primarily in Czech and German, spanning from 1485 to the present, focusing on the late 19th and early 20th centuries. The dataset is designed to support research in document layout analysis and object detection. Each page is annotated with axis-aligned bounding boxes (AABB) representing elements of 25 categories of non-textual elements, such as images, maps, decorative elements, or charts, following the Czech Methodology of image document processing. The annotations were created by expert librarians to ensure accuracy and consistency. The dataset also incorporates pages from multiple, mainly historical, document datasets to enhance variability and maintain continuity. The dataset is divided into development and test subsets, with the test set carefully selected to maintain the category distribution. We provide baseline results using YOLO and DETR object detectors, offering a reference point for future research. The AnnoPage Dataset is publicly available on Zenodo (this https URL), along with ground-truth annotations in YOLO format. 

**Abstract (ZH)**: AnnoPage 数据集：一种包含从 1485 年至今历史文献中的 7550 页文档的新颖集合，主要使用捷克语和德语文本，重点关注 19 世纪晚期和 20 世纪早期。该数据集旨在支持文档布局分析和对象检测研究。每页都标注有表示 25 类非文本元素（如图像、地图、装饰元素或图表）的轴对齐边界框（AABB），遵循捷克图像文档处理方法学。标注工作由专家图书馆员完成，以确保准确性和一致性。数据集还整合了多个主要来自历史文献的数据集，以增强多样性和保持连贯性。数据集分为开发集和测试集，测试集精心挑选以保持类别分布的一致性。我们提供了使用 YOLO 和 DETR 对象检测器的基准结果，为未来的研究提供参考点。AnnoPage 数据集可在 Zenodo 上公开获取 (这个 https URL)，并提供 YOLO 格式的 ground-truth 注释。 

---
# Robust Offline Imitation Learning Through State-level Trajectory Stitching 

**Title (ZH)**: 基于状态级轨迹拼接的鲁棒离线 imitation 学习 

**Authors**: Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22524)  

**Abstract**: Imitation learning (IL) has proven effective for enabling robots to acquire visuomotor skills through expert demonstrations. However, traditional IL methods are limited by their reliance on high-quality, often scarce, expert data, and suffer from covariate shift. To address these challenges, recent advances in offline IL have incorporated suboptimal, unlabeled datasets into the training. In this paper, we propose a novel approach to enhance policy learning from mixed-quality offline datasets by leveraging task-relevant trajectory fragments and rich environmental dynamics. Specifically, we introduce a state-based search framework that stitches state-action pairs from imperfect demonstrations, generating more diverse and informative training trajectories. Experimental results on standard IL benchmarks and real-world robotic tasks showcase that our proposed method significantly improves both generalization and performance. 

**Abstract (ZH)**: 利用与任务相关的轨迹片段和丰富的环境动力学增强低质量离线数据的策略学习 

---
# Exploiting Mixture-of-Experts Redundancy Unlocks Multimodal Generative Abilities 

**Title (ZH)**: 充分利用Mixture-of-Experts冗余性解锁多模态生成能力 

**Authors**: Raman Dutt, Harleen Hanspal, Guoxuan Xia, Petru-Daniel Tudosiu, Alexander Black, Yongxin Yang, Steven McDonagh, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2503.22517)  

**Abstract**: In this work, we undertake the challenge of augmenting the existing generative capabilities of pre-trained text-only large language models (LLMs) with multi-modal generation capability while satisfying two core constraints: C1 preserving the preservation of original language generative capabilities with negligible performance degradation, and C2 adhering to a small parameter budget to learn the new modality, ensuring scalability and efficiency. In contrast to current approaches that add dedicated modules, thereby significantly increasing the parameter count, we propose a method that leverages the underutilized capacity inherent in deep models. Specifically, we exploit the parameter redundancy within Mixture-of-Experts (MoEs) as a source of additional capacity for learning a new modality, enabling better parameter efficiency (C1). Moreover, we preserve the original language generation capabilities by applying low-rank adaptation exclusively to the tokens of the new modality (C2). Furthermore, we introduce a novel parameter initialization scheme based on the Gromov-Wasserstein distance to improve convergence and training stability. Through an extensive analysis of the routing mechanism, we uncover the emergence of modality-specific pathways and decreased redundancy within the experts that can efficiently unlock multi-modal generative capabilities. Overall, our method can be seamlessly applied to a wide range of contemporary LLMs, providing a new pathway for transitioning from uni-modal to multi-modal architectures. 

**Abstract (ZH)**: 本研究致力于在保留原有语言生成能力不明显下降的前提下，增强预训练文本型大型语言模型（LLMs）的多模态生成能力，同时遵守两项核心约束：C1保持原始语言生成能力，C2保持参数预算小，确保可扩展性和效率。与当前通过添加专用模块来显著增加参数数量的方法不同，我们提出了一种利用深层模型中未充分利用的能力的方法。具体而言，我们利用Mixture-of-Experts（MoEs）内的参数冗余作为学习新模态的额外能力来源，实现更好的参数效率（C1）。同时，我们通过仅对新模态的标记应用低秩适应来保留原始语言生成能力（C2）。此外，我们提出了基于Gromov-Wasserstein距离的新参数初始化方案，以提高收敛性和训练稳定性。通过广泛分析路由机制，我们发现模态特定路径的出现和专家内冗余度的减少，可以高效地解锁多模态生成能力。总体而言，我们的方法可以无缝应用于广泛 contemporary LLMs，提供了一条从单模态向多模态架构过渡的新途径。 

---
# Masked Self-Supervised Pre-Training for Text Recognition Transformers on Large-Scale Datasets 

**Title (ZH)**: 大规模数据集上的掩蔽自我监督预训练文本识别变换器 

**Authors**: Martin Kišš, Michal Hradiš  

**Link**: [PDF](https://arxiv.org/pdf/2503.22513)  

**Abstract**: Self-supervised learning has emerged as a powerful approach for leveraging large-scale unlabeled data to improve model performance in various domains. In this paper, we explore masked self-supervised pre-training for text recognition transformers. Specifically, we propose two modifications to the pre-training phase: progressively increasing the masking probability, and modifying the loss function to incorporate both masked and non-masked patches. We conduct extensive experiments using a dataset of 50M unlabeled text lines for pre-training and four differently sized annotated datasets for fine-tuning. Furthermore, we compare our pre-trained models against those trained with transfer learning, demonstrating the effectiveness of the self-supervised pre-training. In particular, pre-training consistently improves the character error rate of models, in some cases up to 30 % relatively. It is also on par with transfer learning but without relying on extra annotated text lines. 

**Abstract (ZH)**: 自监督学习作为一种利用大规模未标注数据提高各种领域模型性能的强大方法已经Emerged。本文我们探索文本识别变换器的掩码自监督预训练方法。具体而言，我们在预训练阶段提出了两种修改：逐步增加掩码概率，并修改损失函数以同时考虑掩码和非掩码片段。我们使用包含50百万个未标注文本行的数据集进行预训练，并使用四个不同大小的标注数据集进行微调。此外，我们还将我们的预训练模型与通过迁移学习训练的模型进行比较，以证明自监督预训练的有效性。特别是，预训练一致地提高了字符错误率，某些情况下相对提高了多达30%。另一方面，它的效果与迁移学习相当，但无需依赖额外的标注文本行。 

---
# Almost Bayesian: The Fractal Dynamics of Stochastic Gradient Descent 

**Title (ZH)**: 几乎贝叶斯：随机梯度下降的分形动态 

**Authors**: Max Hennick, Stijn De Baerdemacker  

**Link**: [PDF](https://arxiv.org/pdf/2503.22478)  

**Abstract**: We show that the behavior of stochastic gradient descent is related to Bayesian statistics by showing that SGD is effectively diffusion on a fractal landscape, where the fractal dimension can be accounted for in a purely Bayesian way. By doing this we show that SGD can be regarded as a modified Bayesian sampler which accounts for accessibility constraints induced by the fractal structure of the loss landscape. We verify our results experimentally by examining the diffusion of weights during training. These results offer insight into the factors which determine the learning process, and seemingly answer the question of how SGD and purely Bayesian sampling are related. 

**Abstract (ZH)**: 我们展示了随机梯度下降的行为与贝叶斯统计之间的关系，通过证明SGD实际上是在分形景观上的扩散过程，分形维数可以用纯粹的贝叶斯方法来解释。通过这种方式，我们表明SGD可以被视为一种修正过的贝叶斯采样器，它考虑了由损失景观分形结构引起的可访问性约束。我们通过检查训练过程中权重的扩散来实验验证这些结果。这些结果为我们提供了决定学习过程的因素提供了见解，并似乎回答了SGD和纯粹贝叶斯采样之间关系的问题。 

---
# Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey 

**Title (ZH)**: 基于LLM的代理在多轮对话中的评估：一个综述 

**Authors**: Shengyue Guan, Haoyi Xiong, Jindong Wang, Jiang Bian, Bin Zhu, Jian-guang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22458)  

**Abstract**: This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues. 

**Abstract (ZH)**: 本调研考察基于大型语言模型（LLM）的代理在多轮对话设置中的评估方法。我们采用了借鉴PRISMA框架的方法，系统地回顾了近250篇学术资源，涵盖了各种出版平台的前沿状态，并为我们的分析奠定了坚实的基础。本研究通过开发两个相互关联的分类系统提供了一种结构化的方法：一个定义了“评估什么”，另一个解释了“如何评估”。第一个分类系统识别了基于LLM的代理在多轮对话中的关键组件及其评估维度，包括任务完成、响应质量、用户体验、记忆和语境保留，以及规划和工具集成。这些组件确保了对话代理的性能评估是全面且有意义的。第二个分类系统专注于评估方法。它将方法归类为人标注评估、自动指标、将人类评估与定量指标相结合的混合策略，以及利用LLM进行自我评判的方法。该框架不仅捕捉了传统的语言理解指标，如BLEU和ROUGE分数，还纳入了反映多轮对话动态互动性质的先进技术。 

---
# Entropy-guided sequence weighting for efficient exploration in RL-based LLM fine-tuning 

**Title (ZH)**: 基于RL的LLM微调中熵导向的序列加权高效探索方法 

**Authors**: Abdullah Vanlioglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22456)  

**Abstract**: We introduce Entropy-Guided Sequence Weighting (EGSW), a novel approach that enhances the exploration-exploitation tradeoff by dynamically assigning weights to generated outputs based on their advantage and entropy for Reinforcement Learning-based Large Language Model fine-tuning. EGSW integrates entropy regularization with advantage-based weighting to balance policy updates, enabling efficient exploration in high-dimensional state spaces. By employing temperature-scaled softmax weighting over sequences, EGSW prioritizing high-reward, high-uncertainty steps while maintaining training stability. Although originally developed to improve Group Relative Policy Optimization (GRPO) during large language model (LLM) fine-tuning, EGSW is generalizable to other reinforcement learning (RL) algorithms and can be implemented in both step-wise and trajectory-wise settings. Empirical evaluations demonstrate that EGSW enhances GRPO reasoning ability, yielding improvements in sample efficiency. Future work will explore the application of EGSW to advanced RL methodologies. 

**Abstract (ZH)**: 熵导向序列权重分配（EGSW）在强化学习导向的大语言模型微调中的探索与利用权衡增强方法 

---
# A Causal Framework to Measure and Mitigate Non-binary Treatment Discrimination 

**Title (ZH)**: 一种用于测量和减轻非二元治疗歧视的因果框架 

**Authors**: Ayan Majumdar, Deborah D. Kanubala, Kavya Gupta, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2503.22454)  

**Abstract**: Fairness studies of algorithmic decision-making systems often simplify complex decision processes, such as bail or loan approvals, into binary classification tasks. However, these approaches overlook that such decisions are not inherently binary (e.g., approve or not approve bail or loan); they also involve non-binary treatment decisions (e.g., bail conditions or loan terms) that can influence the downstream outcomes (e.g., loan repayment or reoffending). In this paper, we argue that non-binary treatment decisions are integral to the decision process and controlled by decision-makers and, therefore, should be central to fairness analyses in algorithmic decision-making. We propose a causal framework that extends fairness analyses and explicitly distinguishes between decision-subjects' covariates and the treatment decisions. This specification allows decision-makers to use our framework to (i) measure treatment disparity and its downstream effects in historical data and, using counterfactual reasoning, (ii) mitigate the impact of past unfair treatment decisions when automating decision-making. We use our framework to empirically analyze four widely used loan approval datasets to reveal potential disparity in non-binary treatment decisions and their discriminatory impact on outcomes, highlighting the need to incorporate treatment decisions in fairness assessments. Moreover, by intervening in treatment decisions, we show that our framework effectively mitigates treatment discrimination from historical data to ensure fair risk score estimation and (non-binary) decision-making processes that benefit all stakeholders. 

**Abstract (ZH)**: 算法决策系统中的公平性研究往往将复杂的决策过程（如保释或贷款审批）简化为二元分类任务。然而，这些方法忽视了这些决策本质上并非二元的（例如，批准或不批准保释或贷款），还涉及影响下游结果（如贷款偿还或重新犯罪）的非二元治疗决策。在本文中，我们主张非二元治疗决策是决策过程的重要组成部分，由决策者控制，因此在算法决策中的公平性分析中应占据中心地位。我们提出了一种因果框架，将其纳入公平性分析中，并明确区分决策主体的协变量和治疗决策。这种建模允许决策者使用我们的框架（i）在历史数据中衡量治疗差异及其下游影响，并利用反事实推理（ii）在自动化决策时减轻过去不公平治疗决策的影响。我们使用我们的框架对四个广泛使用的贷款审批数据集进行实证分析，揭示了非二元治疗决策中潜在的差异及其对结果的歧视性影响，强调了在公平评估中纳入治疗决策的必要性。此外，通过干预治疗决策，我们展示了我们的框架如何有效地从历史数据中缓解治疗歧视，以确保公平的风险评分估计和（非二元）决策过程，从而使所有相关方受益。 

---
# CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching 

**Title (ZH)**: CoSIL: 软件问题定位 via LLM 驱动的代码仓库图搜索 

**Authors**: Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22424)  

**Abstract**: Large language models (LLMs) have significantly advanced autonomous software engineering, leading to a growing number of software engineering agents that assist developers in automatic program repair. Issue localization forms the basis for accurate patch generation. However, because of limitations caused by the context window length of LLMs, existing issue localization methods face challenges in balancing concise yet effective contexts and adequately comprehensive search spaces. In this paper, we introduce CoSIL, an LLM driven, simple yet powerful function level issue localization method without training or indexing. CoSIL reduces the search space through module call graphs, iteratively searches the function call graph to obtain relevant contexts, and uses context pruning to control the search direction and manage contexts effectively. Importantly, the call graph is dynamically constructed by the LLM during search, eliminating the need for pre-parsing. Experiment results demonstrate that CoSIL achieves a Top-1 localization success rate of 43 percent and 44.6 percent on SWE bench Lite and SWE bench Verified, respectively, using Qwen2.5 Coder 32B, outperforming existing methods by 8.6 to 98.2 percent. When CoSIL is applied to guide the patch generation stage, the resolved rate further improves by 9.3 to 31.5 percent. 

**Abstract (ZH)**: 大型语言模型（LLMs）显著推动了自主软件工程的发展，导致出现越来越多的软件工程代理，协助开发人员进行自动程序修复。问题定位是准确生成补丁的基础。然而，由于受限于LLMs的上下文窗口长度，现有的问题定位方法在简洁有效的情景描述和足够全面的搜索空间之间面临着平衡挑战。在本文中，我们介绍了CoSIL，这是一种由LLMs驱动、简单高效且无需训练或索引的功能级别问题定位方法。CoSIL通过模块调用图减少搜索空间，迭代搜索函数调用图以获得相关上下文，并使用上下文修剪来控制搜索方向并有效地管理上下文。重要的是，调用图在搜索过程中由LLMs动态构建，消除了预先解析的需求。实验结果表明，使用Qwen2.5 Coder 32B，CoSIL在SWE bench Lite和SWE bench Verified上的Top-1定位成功率分别为43%和44.6%，优于现有方法8.6%至98.2%。当CoSIL应用于指导补丁生成阶段时，解决率进一步提高9.3%至31.5%。 

---
# Training Large Language Models for Advanced Typosquatting Detection 

**Title (ZH)**: 训练大规模语言模型以实现高级.typo squatting检测 

**Authors**: Jackson Welch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22406)  

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection. 

**Abstract (ZH)**: typosquatting是长期存在的网络威胁，通过利用用户在输入URL时的错误来欺骗用户、分发恶意软件并进行钓鱼攻击。随着域名和新的顶级域名(TLDs)的增多，typosquatting技术日益成熟，对个人、企业和国家的网络安全基础设施构成了重大风险。传统检测方法主要侧重于已知的仿冒模式，难以识别更复杂的攻击。本文提出了一种新的方法，利用大规模语言模型(LLMs)增强typosquatting检测。通过在字符级转换和基于模式的启发式规则上训练LLM，而不是特定于域的数据，开发出一种更加适应和稳健的检测机制。实验结果显示，Phi-4 14B模型在适当微调后，仅使用少量训练样本就实现了98%的准确率。该研究强调了LLMs在网络安全领域应用的潜力，特别是在缓解基于域的欺骗战术方面，并提供了优化机器学习策略以提高威胁检测效果的见解。 

---
# EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing 

**Title (ZH)**: EllieSQL: 基于复杂性意识路由的低成本文本到SQL转换 

**Authors**: Yizhang Zhu, Runzhi Jiang, Boyan Li, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22402)  

**Abstract**: Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs--often overlooked--stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2x boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL. 

**Abstract (ZH)**: 复杂性意识路由框架EllieSQL：基于估计复杂性的查询路由以提升成本效率的Text-to-SQL 

---
# On-site estimation of battery electrochemical parameters via transfer learning based physics-informed neural network approach 

**Title (ZH)**: 基于物理约束神经网络的迁移学习在现场估计电池电化学参数 

**Authors**: Josu Yeregui, Iker Lopetegi, Sergio Fernandez, Erik Garayalde, Unai Iraola  

**Link**: [PDF](https://arxiv.org/pdf/2503.22396)  

**Abstract**: This paper presents a novel physical parameter estimation framework for on-site model characterization, using a two-phase modelling strategy with Physics-Informed Neural Networks (PINNs) and transfer learning (TL). In the first phase, a PINN is trained using only the physical principles of the single particle model (SPM) equations. In the second phase, the majority of the PINN parameters are frozen, while critical electrochemical parameters are set as trainable and adjusted using real-world voltage profile data. The proposed approach significantly reduces computational costs, making it suitable for real-time implementation on Battery Management Systems (BMS). Additionally, as the initial phase does not require field data, the model is easy to deploy with minimal setup requirements. With the proposed methodology, we have been able to effectively estimate relevant electrochemical parameters with operating data. This has been proved estimating diffusivities and active material volume fractions with charge data in different degradation conditions. The methodology is experimentally validated in a Raspberry Pi device using data from a standard charge profile with a 3.89\% relative accuracy estimating the active material volume fractions of a NMC cell with 82.09\% of its nominal capacity. 

**Abstract (ZH)**: 本文提出了一种用于现场模型表征的新型物理参数估计框架，采用包含物理学知识的人工神经网络（PINNs）和迁移学习（TL）的两阶段建模策略。在第一阶段，仅使用单粒子模型（SPM）方程的物理原理训练PINN。在第二阶段，固定PINN的大部分参数，将关键电化学参数设为可训练，并使用实际电压轮廓数据进行调整。所提出的方法显著降低了计算成本，使其适用于电池管理系统（BMS）的实时实现。此外，由于初始阶段不需要现场数据，该模型部署简单，无需复杂设置。通过所提出的方法，我们能够利用运行数据有效地估计相关电化学参数，证明了在不同退化条件下利用充放电数据估计扩散系数和活性材料体积分数的有效性。该方法在基于树莓派的设备上使用标准充放电数据集进行了实验验证，相对准确度为3.89%，估计镍锰钴（NMC）电池在82.09%额定容量时的活性材料体积分数。 

---
# Endo-TTAP: Robust Endoscopic Tissue Tracking via Multi-Facet Guided Attention and Hybrid Flow-point Supervision 

**Title (ZH)**: 内窥镜组织跟踪器：基于多面引导注意力和混合流点监督的稳健内窥镜组织跟踪 

**Authors**: Rulin Zhou, Wenlong He, An Wang, Qiqi Yao, Haijun Hu, Jiankun Wang, Xi Zhang an Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.22394)  

**Abstract**: Accurate tissue point tracking in endoscopic videos is critical for robotic-assisted surgical navigation and scene understanding, but remains challenging due to complex deformations, instrument occlusion, and the scarcity of dense trajectory annotations. Existing methods struggle with long-term tracking under these conditions due to limited feature utilization and annotation dependence. We present Endo-TTAP, a novel framework addressing these challenges through: (1) A Multi-Facet Guided Attention (MFGA) module that synergizes multi-scale flow dynamics, DINOv2 semantic embeddings, and explicit motion patterns to jointly predict point positions with uncertainty and occlusion awareness; (2) A two-stage curriculum learning strategy employing an Auxiliary Curriculum Adapter (ACA) for progressive initialization and hybrid supervision. Stage I utilizes synthetic data with optical flow ground truth for uncertainty-occlusion regularization, while Stage II combines unsupervised flow consistency and semi-supervised learning with refined pseudo-labels from off-the-shelf trackers. Extensive validation on two MICCAI Challenge datasets and our collected dataset demonstrates that Endo-TTAP achieves state-of-the-art performance in tissue point tracking, particularly in scenarios characterized by complex endoscopic conditions. The source code and dataset will be available at this https URL. 

**Abstract (ZH)**: 精确的内窥镜视频组织点跟踪对于机器人辅助手术导航和场景理解至关重要，但由于复杂的变形、器械遮挡和密集轨迹注解稀缺的限制，这一任务仍然具有挑战性。现有方法在这些条件下难以实现长期跟踪，因为它们在特征利用和注解依赖方面存在局限。我们提出了Endo-TTAP，这是一种通过以下方式解决这些挑战的新框架：(1) 多尺度流动态、DINOv2语义嵌入和显式运动模式协同作用的多面引导注意力(MFGA)模块，用于联合预测点位置及其不确定性和遮挡感知；(2) 采用辅助课程适配器(ACA)的两阶段课程学习策略，实现渐进初始化和混合监督。第一阶段利用具有光学流 ground truth 的合成数据进行不确定性和遮挡正则化，而第二阶段结合了无监督流一致性监督与改进的离线成品跟踪器伪标签的半监督学习。在两个MICCAI挑战数据集和我们收集的数据集上的广泛验证表明，Endo-TTAP 在组织点跟踪方面达到了最先进的性能，特别是在复杂内窥镜条件下。源代码和数据集将在此处提供。 

---
# ViSketch-GPT: Collaborative Multi-Scale Feature Extraction for Sketch Recognition and Generation 

**Title (ZH)**: ViSketch-GPT: 协同多尺度特征提取的草图识别与生成 

**Authors**: Giulio Federico, Giuseppe Amato, Fabio Carrara, Claudio Gennaro, Marco Di Benedetto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22374)  

**Abstract**: Understanding the nature of human sketches is challenging because of the wide variation in how they are created. Recognizing complex structural patterns improves both the accuracy in recognizing sketches and the fidelity of the generated sketches. In this work, we introduce ViSketch-GPT, a novel algorithm designed to address these challenges through a multi-scale context extraction approach. The model captures intricate details at multiple scales and combines them using an ensemble-like mechanism, where the extracted features work collaboratively to enhance the recognition and generation of key details crucial for classification and generation tasks.
The effectiveness of ViSketch-GPT is validated through extensive experiments on the QuickDraw dataset. Our model establishes a new benchmark, significantly outperforming existing methods in both classification and generation tasks, with substantial improvements in accuracy and the fidelity of generated sketches.
The proposed algorithm offers a robust framework for understanding complex structures by extracting features that collaborate to recognize intricate details, enhancing the understanding of structures like sketches and making it a versatile tool for various applications in computer vision and machine learning. 

**Abstract (ZH)**: 理解人类草图的本质因创作方式的广泛差异而具有挑战性。识别复杂的结构模式可以提高草图识别的准确性和生成草图的保真度。在本工作中，我们提出了一种名为ViSketch-GPT的新型算法，通过多尺度上下文提取方法来应对这些挑战。该模型在多尺度上捕捉复杂的细微特征，并通过类似集成的机制将它们结合在一起，提取的特征协同工作以增强关键细节的识别和生成，这些细节对于分类和生成任务至关重要。

ViSketch-GPT的有效性通过在QuickDraw数据集上的大量实验得到验证。我们的模型在分类和生成任务中均建立了新的基准，显著优于现有方法，在准确性和生成草图的保真度方面取得了显着改进。

提出的算法提供了一种稳健的框架，用于通过提取协同工作的特征来理解复杂结构，增强了对如草图等结构的理解，并使其成为计算机视觉和机器学习中各种应用的多功能工具。 

---
# ForcePose: A Deep Learning Approach for Force Calculation Based on Action Recognition Using MediaPipe Pose Estimation Combined with Object Detection 

**Title (ZH)**: 基于MediaPipe姿态估计与物体检测结合的动作识别的力计算深度学习方法：ForcePose 

**Authors**: Nandakishor M, Vrinda Govind V, Anuradha Puthalath, Anzy L, Swathi P S, Aswathi R, Devaprabha A R, Varsha Raj, Midhuna Krishnan K, Akhila Anilkumar T V, Yamuna P V  

**Link**: [PDF](https://arxiv.org/pdf/2503.22363)  

**Abstract**: Force estimation in human-object interactions is crucial for various fields like ergonomics, physical therapy, and sports science. Traditional methods depend on specialized equipment such as force plates and sensors, which makes accurate assessments both expensive and restricted to laboratory settings. In this paper, we introduce ForcePose, a novel deep learning framework that estimates applied forces by combining human pose estimation with object detection. Our approach leverages MediaPipe for skeletal tracking and SSD MobileNet for object recognition to create a unified representation of human-object interaction. We've developed a specialized neural network that processes both spatial and temporal features to predict force magnitude and direction without needing any physical sensors. After training on our dataset of 850 annotated videos with corresponding force measurements, our model achieves a mean absolute error of 5.83 N in force magnitude and 7.4 degrees in force direction. When compared to existing computer vision approaches, our method performs 27.5% better while still offering real-time performance on standard computing hardware. ForcePose opens up new possibilities for force analysis in diverse real-world scenarios where traditional measurement tools are impractical or intrusive. This paper discusses our methodology, the dataset creation process, evaluation metrics, and potential applications across rehabilitation, ergonomics assessment, and athletic performance analysis. 

**Abstract (ZH)**: 人体与物体交互中的力估计对于人机工程学、物理治疗和运动科学等领域至关重要。传统方法依赖于力板和传感器等专用设备，这使得准确评估既昂贵又局限于实验室环境。本文介绍了一种新颖的深度学习框架ForcePose，通过结合人体姿态估计和物体检测来估计施加的力。我们的方法利用MediaPipe进行骨骼跟踪，SSD-MobileNet进行物体识别，从而创建人体-物体交互的统一表示。我们开发了一个专用的神经网络，处理空间和时间特征以预测力的大小和方向，而无需任何物理传感器。在包含850个标注视频及其相应力测量值的训练集上训练后，我们的模型在力的大小上实现了5.83 N的平均绝对误差，在力的方向上实现了7.4度的误差。与现有的计算机视觉方法相比，我们的方法在标准计算硬件上仍能实现实时性能，且性能提高了27.5%。ForcePose为在传统测量工具不切实际或侵入性的多种现实场景中进行力分析开辟了新途径。本文讨论了我们的方法论、数据集创建过程、评估指标以及在康复、人机工程学评估和运动表现分析中的潜在应用。 

---
# Shapley Revisited: Tractable Responsibility Measures for Query Answers 

**Title (ZH)**: Shapley值重探：查询答案的责任度量方法 

**Authors**: Meghyn Bienvenu, Diego Figueira, Pierre Lafourcade  

**Link**: [PDF](https://arxiv.org/pdf/2503.22358)  

**Abstract**: The Shapley value, originating from cooperative game theory, has been employed to define responsibility measures that quantify the contributions of database facts to obtaining a given query answer. For non-numeric queries, this is done by considering a cooperative game whose players are the facts and whose wealth function assigns 1 or 0 to each subset of the database, depending on whether the query answer holds in the given subset. While conceptually simple, this approach suffers from a notable drawback: the problem of computing such Shapley values is #P-hard in data complexity, even for simple conjunctive queries. This motivates us to revisit the question of what constitutes a reasonable responsibility measure and to introduce a new family of responsibility measures -- weighted sums of minimal supports (WSMS) -- which satisfy intuitive properties. Interestingly, while the definition of WSMSs is simple and bears no obvious resemblance to the Shapley value formula, we prove that every WSMS measure can be equivalently seen as the Shapley value of a suitably defined cooperative game. Moreover, WSMS measures enjoy tractable data complexity for a large class of queries, including all unions of conjunctive queries. We further explore the combined complexity of WSMS computation and establish (in)tractability results for various subclasses of conjunctive queries. 

**Abstract (ZH)**: 基于 cooperaive 博弈理论的 Shapley 值已被用于定义衡量数据库事实对获得给定查询答案的贡献的责任度量。对于非数值查询，通过考虑玩家为数据库事实且财富函数对数据库子集分配 1 或 0 的博弈来进行。尽管概念上简单，但这种方法存在明显的缺点：计算此类 Shapley 值在数据复杂性上是 #P-难问题，即使是简单的合取查询也是如此。这促使我们重新审视什么是合理的责任度量，并引入一种新的责任度量家族——最小支持加权和（WSMS），它们满足直观的性质。有趣的是，尽管 WSMS 的定义简单且与 Shapley 值公式无明显联系，我们证明每个 WSMS 度量都可以等价地视为适当定义的博弈的 Shapley 值。此外，对于一类广泛的查询（包括所有合取查询的并集），WSMS 度量在数据复杂性上是可处理的。我们进一步探讨了 WSMS 计算的组合复杂性，并为多种合取查询的子类建立了可处理性和不可处理性结果。 

---
# Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions 

**Title (ZH)**: 企业还是多变？评估大规模语言模型在序列交互中的一致性 

**Authors**: Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2503.22353)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various tasks, but their deployment in high-stake domains requires consistent performance across multiple interaction rounds. This paper introduces a comprehensive framework for evaluating and improving LLM response consistency, making three key contributions. First, we propose a novel Position-Weighted Consistency (PWC) score that captures both the importance of early-stage stability and recovery patterns in multi-turn interactions. Second, we present a carefully curated benchmark dataset spanning diverse domains and difficulty levels, specifically designed to evaluate LLM consistency under various challenging follow-up scenarios. Third, we introduce Confidence-Aware Response Generation (CARG), a framework that significantly improves response stability by incorporating model confidence signals into the generation process. Empirical results demonstrate that CARG significantly improves response stability without sacrificing accuracy, underscoring its potential for reliable LLM deployment in critical applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中展现出了卓越的能力，但在高风险领域中的部署需要其在多轮交互中保持一致的性能。本文提出了一个全面的框架来评估和提高LLM响应一致性，作出了三项关键贡献。首先，我们提出了一种新型的位置加权一致性（PWC）评分，以捕捉多轮交互中早期稳定性及恢复模式的重要性。其次，我们提供了一个精心策划的基准数据集，覆盖了多个领域和难度级别，旨在评估LLM在各种具有挑战性的后续场景中的一致性。第三，我们引入了一种基于置信度的响应生成（CARG）框架，通过将模型置信度信号集成到生成过程中，显著提高了响应稳定性。实证结果表明，CARG在不牺牲准确性的情况下显著提高了响应稳定性，突显了其在关键应用中可靠部署的潜力。 

---
# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow 

**Title (ZH)**: VoteFlow: 强化自监督场景流中的局部刚性约束 

**Authors**: Yancong Lin, Shiming Wang, Liangliang Nan, Julian Kooij, Holger Caesar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22328)  

**Abstract**: Scene flow estimation aims to recover per-point motion from two adjacent LiDAR scans. However, in real-world applications such as autonomous driving, points rarely move independently of others, especially for nearby points belonging to the same object, which often share the same motion. Incorporating this locally rigid motion constraint has been a key challenge in self-supervised scene flow estimation, which is often addressed by post-processing or appending extra regularization. While these approaches are able to improve the rigidity of predicted flows, they lack an architectural inductive bias for local rigidity within the model structure, leading to suboptimal learning efficiency and inferior performance. In contrast, we enforce local rigidity with a lightweight add-on module in neural network design, enabling end-to-end learning. We design a discretized voting space that accommodates all possible translations and then identify the one shared by nearby points by differentiable voting. Additionally, to ensure computational efficiency, we operate on pillars rather than points and learn representative features for voting per pillar. We plug the Voting Module into popular model designs and evaluate its benefit on Argoverse 2 and Waymo datasets. We outperform baseline works with only marginal compute overhead. Code is available at this https URL. 

**Abstract (ZH)**: 场景流估计旨在从两个相邻的LiDAR扫描中恢复每个点的运动。然而，在自动驾驶等实际应用中，点之间通常不是独立移动的，尤其是属于同一物体的附近点，它们往往共享相同的运动。将这种局部刚性运动约束纳入自监督场景流估计中一直是一个关键挑战，通常通过后处理或附加额外正则化来解决。虽然这些方法能够提高预测流的刚性，但它们缺乏用于局部刚性的架构归纳偏差，导致学习效率低下且性能欠佳。相比之下，我们通过神经网络设计中的轻量级附加模块来强制实施局部刚性，从而实现端到端学习。我们设计了一个离散化投票空间，包含所有可能的平移，并通过可微投票识别附近点共有的平移。此外，为了确保计算效率，我们基于柱状结构进行操作，并为每个柱状结构学习代表特征用于投票。我们将投票模块插入流行模型设计中，并在Argoverse 2和Waymo数据集上评估其益处。我们仅以微小的计算成本超出了基线工作。代码可在以下链接获取。 

---
# AH-GS: Augmented 3D Gaussian Splatting for High-Frequency Detail Representation 

**Title (ZH)**: AH-GS: 增强的3D高斯点云表示方法用于高频细节表达 

**Authors**: Chenyang Xu, XingGuo Deng, Rui Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22324)  

**Abstract**: The 3D Gaussian Splatting (3D-GS) is a novel method for scene representation and view synthesis. Although Scaffold-GS achieves higher quality real-time rendering compared to the original 3D-GS, its fine-grained rendering of the scene is extremely dependent on adequate viewing angles. The spectral bias of neural network learning results in Scaffold-GS's poor ability to perceive and learn high-frequency information in the scene. In this work, we propose enhancing the manifold complexity of input features and using network-based feature map loss to improve the image reconstruction quality of 3D-GS models. We introduce AH-GS, which enables 3D Gaussians in structurally complex regions to obtain higher-frequency encodings, allowing the model to more effectively learn the high-frequency information of the scene. Additionally, we incorporate high-frequency reinforce loss to further enhance the model's ability to capture detailed frequency information. Our result demonstrates that our model significantly improves rendering fidelity, and in specific scenarios (e.g., MipNeRf360-garden), our method exceeds the rendering quality of Scaffold-GS in just 15K iterations. 

**Abstract (ZH)**: 3D高斯喷涂增强的复杂流形输入特征和基于网络的特征图损失在场景表示和视图合成中的应用 

---
# Machine Learning Models for Soil Parameter Prediction Based on Satellite, Weather, Clay and Yield Data 

**Title (ZH)**: 基于卫星、气象、粘土和产量数据的土壤参数预测机器学习模型 

**Authors**: Calvin Kammerlander, Viola Kolb, Marinus Luegmair, Lou Scheermann, Maximilian Schmailzl, Marco Seufert, Jiayun Zhang, Denis Dalic, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2503.22276)  

**Abstract**: Efficient nutrient management and precise fertilization are essential for advancing modern agriculture, particularly in regions striving to optimize crop yields sustainably. The AgroLens project endeavors to address this challenge by develop ing Machine Learning (ML)-based methodologies to predict soil nutrient levels without reliance on laboratory tests. By leveraging state of the art techniques, the project lays a foundation for acionable insights to improve agricultural productivity in resource-constrained areas, such as Africa. The approach begins with the development of a robust European model using the LUCAS Soil dataset and Sentinel-2 satellite imagery to estimate key soil properties, including phosphorus, potassium, nitrogen, and pH levels. This model is then enhanced by integrating supplementary features, such as weather data, harvest rates, and Clay AI-generated embeddings. This report details the methodological framework, data preprocessing strategies, and ML pipelines employed in this project. Advanced algorithms, including Random Forests, Extreme Gradient Boosting (XGBoost), and Fully Connected Neural Networks (FCNN), were implemented and finetuned for precise nutrient prediction. Results showcase robust model performance, with root mean square error values meeting stringent accuracy thresholds. By establishing a reproducible and scalable pipeline for soil nutrient prediction, this research paves the way for transformative agricultural applications, including precision fertilization and improved resource allocation in underresourced regions like Africa. 

**Abstract (ZH)**: 高效的养分管理与精确施肥对于推动现代农业的发展至关重要，特别是在寻求可持续优化作物产量的地区。AgroLens项目旨在通过开发基于机器学习（ML）的方法来预测土壤养分含量，从而解决这一挑战，无需依赖实验室测试。该项目借助先进的技术为基础，为资源受限地区（如非洲）提供可采取的见解，以提高农业生产效率奠定基础。该方法首先利用LUCAS土壤数据集和Sentinel-2卫星影像开发了一个稳健的欧洲模型，以估算关键土壤属性，包括磷、钾、氮和pH值。随后通过整合辅助特征，如天气数据、收获率和Clay AI生成的嵌入式特征来增强该模型。本报告详细介绍了该项目的方法论框架、数据预处理策略和所使用的ML管道。实现了包括随机森林、极端梯度提升（XGBoost）和全连接神经网络（FCNN）在内的高级算法，并对其进行了精确养分预测的优化。结果表明，模型表现出稳健的性能，均方根误差值达到了严格的准确度要求。通过建立可复制和可扩展的土壤养分预测管道，这项研究为包括精准施肥和提高资源分配效率在内的变革性农业应用铺平了道路，特别是在非洲等资源匮乏地区。 

---
# Make Some Noise: Towards LLM audio reasoning and generation using sound tokens 

**Title (ZH)**: 嘈音相伴：面向大规模语言模型基于声音的推理与生成的研究 

**Authors**: Shivam Mehta, Nebojsa Jojic, Hannes Gamper  

**Link**: [PDF](https://arxiv.org/pdf/2503.22275)  

**Abstract**: Integrating audio comprehension and generation into large language models (LLMs) remains challenging due to the continuous nature of audio and the resulting high sampling rates. Here, we introduce a novel approach that combines Variational Quantization with Conditional Flow Matching to convert audio into ultra-low bitrate discrete tokens of 0.23kpbs, allowing for seamless integration with text tokens in LLMs. We fine-tuned a pretrained text-based LLM using Low-Rank Adaptation (LoRA) to assess its effectiveness in achieving true multimodal capabilities, i.e., audio comprehension and generation. Our tokenizer outperforms a traditional VQ-VAE across various datasets with diverse acoustic events. Despite the substantial loss of fine-grained details through audio tokenization, our multimodal LLM trained with discrete tokens achieves competitive results in audio comprehension with state-of-the-art methods, though audio generation is poor. Our results highlight the need for larger, more diverse datasets and improved evaluation metrics to advance multimodal LLM performance. 

**Abstract (ZH)**: 将音频理解与生成整合到大型语言模型中仍具有挑战性，原因是音频的连续性导致了高采样率。我们提出了一种结合变分量化与条件流动匹配的新方法，将音频转换为超低比特率的离散令牌（0.23kbps），以便无缝集成到大型语言模型中的文本令牌中。我们使用低秩适应（LoRA）fine-tune了一个预训练的文本基于的大规模语言模型，以评估其在实现真正的跨模态能力，即音频理解与生成方面的效果。我们的分词器在各种包含不同声学事件的数据集中优于传统的VQ-VAE。尽管通过音频分词丢失了大量细粒度的细节，但经过离散令牌训练的跨模态大语言模型在音频理解方面达到了与先进方法相当的结果，尽管音频生成效果较差。我们的研究结果强调了需要更大的、更为多样化的数据集和改进的评估指标，以推动跨模态大语言模型性能的提升。 

---
# Beyond the Script: Testing LLMs for Authentic Patient Communication Styles in Healthcare 

**Title (ZH)**: 超越脚本：在医疗健康领域测试LLM以实现真实的患者沟通风格 

**Authors**: Anna Bodonhelyi, Christian Stegemann-Philipps, Alessandra Sonanini, Lea Herschbach, Márton Szép, Anne Herrmann-Werner, Teresa Festl-Wietek, Enkelejda Kasneci, Friederike Holderried  

**Link**: [PDF](https://arxiv.org/pdf/2503.22250)  

**Abstract**: Effective patient communication is pivotal in healthcare, yet traditional medical training often lacks exposure to diverse, challenging interpersonal dynamics. To bridge this gap, this study proposes the use of Large Language Models (LLMs) to simulate authentic patient communication styles, specifically the "accuser" and "rationalizer" personas derived from the Satir model, while also ensuring multilingual applicability to accommodate diverse cultural contexts and enhance accessibility for medical professionals. Leveraging advanced prompt engineering, including behavioral prompts, author's notes, and stubbornness mechanisms, we developed virtual patients (VPs) that embody nuanced emotional and conversational traits. Medical professionals evaluated these VPs, rating their authenticity (accuser: $3.8 \pm 1.0$; rationalizer: $3.7 \pm 0.8$ on a 5-point Likert scale (from one to five)) and correctly identifying their styles. Emotion analysis revealed distinct profiles: the accuser exhibited pain, anger, and distress, while the rationalizer displayed contemplation and calmness, aligning with predefined, detailed patient description including medical history. Sentiment scores (on a scale from zero to nine) further validated these differences in the communication styles, with the accuser adopting negative ($3.1 \pm 0.6$) and the rationalizer more neutral ($4.0 \pm 0.4$) tone. These results underscore LLMs' capability to replicate complex communication styles, offering transformative potential for medical education. This approach equips trainees to navigate challenging clinical scenarios by providing realistic, adaptable patient interactions, enhancing empathy and diagnostic acumen. Our findings advocate for AI-driven tools as scalable, cost-effective solutions to cultivate nuanced communication skills, setting a foundation for future innovations in healthcare training. 

**Abstract (ZH)**: 使用大型语言模型模拟多元挑战性医患沟通模式以提升医疗教育有效性 

---
# WeatherMesh-3: Fast and accurate operational global weather forecasting 

**Title (ZH)**: WeatherMesh-3：快速且准确的全球天气预报 

**Authors**: Haoxing Du, Lyna Kim, Joan Creus-Costa, Jack Michaels, Anuj Shetty, Todd Hutchinson, Christopher Riedel, John Dean  

**Link**: [PDF](https://arxiv.org/pdf/2503.22235)  

**Abstract**: We present WeatherMesh-3 (WM-3), an operational transformer-based global weather forecasting system that improves the state of the art in both accuracy and computational efficiency. We introduce the following advances: 1) a latent rollout that enables arbitrary-length predictions in latent space without intermediate encoding or decoding; and 2) a modular architecture that flexibly utilizes mixed-horizon processors and encodes multiple real-time analyses to create blended initial conditions. WM-3 generates 14-day global forecasts at 0.25-degree resolution in 12 seconds on a single RTX 4090. This represents a >100,000-fold speedup over traditional NWP approaches while achieving superior accuracy with up to 37.7% improvement in RMSE over operational models, requiring only a single consumer-grade GPU for deployment. We aim for WM-3 to democratize weather forecasting by providing an accessible, lightweight model for operational use while pushing the performance boundaries of machine learning-based weather prediction. 

**Abstract (ZH)**: 我们呈现了WeatherMesh-3 (WM-3)，这是一个基于变换器的全球天气预报系统，它在准确性和计算效率上都超越了现有最佳水平。我们引入了以下进步：1) 潜在滚动，可以在潜在空间中进行任意长度的预测，无需中间编码或解码；2) 模块化架构，灵活利用混合视窗处理器，并编码多个实时分析以生成混合初始条件。WM-3在单块RTX 4090gpu上可在12秒内生成0.25度分辨率的14天全球预报。这代表了与传统数值天气预报方法相比超过10万倍的加速，同时在均方根误差(RMSE)上最高可提高37.7%，仅需一台消费者级GPU即可部署。我们旨在通过提供一个易于访问且轻量的模型来推动气象预报的普及，同时推动基于机器学习的天气预报性能边界。 

---
# Process Reward Modeling with Entropy-Driven Uncertainty 

**Title (ZH)**: 熵驱动不确定性下的过程奖励建模 

**Authors**: Lang Cao, Renhong Chen, Yingtian Zou, Chao Peng, Wu Ning, Huacong Xu, Qian Chen, Yuxian Wang, Peishuo Su, Mofan Peng, Zijie Chen, Yitong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22233)  

**Abstract**: This paper presents the Entropy-Driven Unified Process Reward Model (EDU-PRM), a novel framework that approximates state-of-the-art performance in process supervision while drastically reducing training costs. EDU-PRM introduces an entropy-guided dynamic step partitioning mechanism, using logit distribution entropy to pinpoint high-uncertainty regions during token generation dynamically. This self-assessment capability enables precise step-level feedback without manual fine-grained annotation, addressing a critical challenge in process supervision. Experiments on the Qwen2.5-72B model with only 7,500 EDU-PRM-generated training queries demonstrate accuracy closely approximating the full Qwen2.5-72B-PRM (71.1% vs. 71.6%), achieving a 98% reduction in query cost compared to prior methods. This work establishes EDU-PRM as an efficient approach for scalable process reward model training. 

**Abstract (ZH)**: 熵驱动统一过程奖励模型（EDU-PRM）：一种在大幅降低训练成本的同时逼近先进性能的新框架 

---
# MFH: A Multi-faceted Heuristic Algorithm Selection Approach for Software Verification 

**Title (ZH)**: MFH：软件验证中多面向启发式算法选择方法 

**Authors**: Jie Su, Liansai Deng, Cheng Wen, Rong Wang, Zhi Ma, Nan Zhang, Cong Tian, Zhenhua Duan, Shengchao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22228)  

**Abstract**: Currently, many verification algorithms are available to improve the reliability of software systems. Selecting the appropriate verification algorithm typically demands domain expertise and non-trivial manpower. An automated algorithm selector is thus desired. However, existing selectors, either depend on machine-learned strategies or manually designed heuristics, encounter issues such as reliance on high-quality samples with algorithm labels and limited scalability. In this paper, an automated algorithm selection approach, namely MFH, is proposed for software verification. Our approach leverages the heuristics that verifiers producing correct results typically implement certain appropriate algorithms, and the supported algorithms by these verifiers indirectly reflect which ones are potentially applicable. Specifically, MFH embeds the code property graph (CPG) of a semantic-preserving transformed program to enhance the robustness of the prediction model. Furthermore, our approach decomposes the selection task into the sub-tasks of predicting potentially applicable algorithms and matching the most appropriate verifiers. Additionally, MFH also introduces a feedback loop on incorrect predictions to improve model prediction accuracy. We evaluate MFH on 20 verifiers and over 15,000 verification tasks. Experimental results demonstrate the effectiveness of MFH, achieving a prediction accuracy of 91.47% even without ground truth algorithm labels provided during the training phase. Moreover, the prediction accuracy decreases only by 0.84% when introducing 10 new verifiers, indicating the strong scalability of the proposed approach. 

**Abstract (ZH)**: 一种用于软件验证的自动化算法选择方法：MFH 

---
# Learning to Instruct for Visual Instruction Tuning 

**Title (ZH)**: 视觉指令调优的指令学习 

**Authors**: Zhihan Zhou, Feng Hong, Jiaan Luo, Jiangchao Yao, Dongsheng Li, Bo Han, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22215)  

**Abstract**: We propose LIT, an advancement of visual instruction tuning (VIT). While VIT equips Multimodal LLMs (MLLMs) with promising multimodal capabilities, the current design choices for VIT often result in overfitting and shortcut learning, potentially degrading performance. This gap arises from an overemphasis on instruction-following abilities, while neglecting the proactive understanding of visual information. Inspired by this, LIT adopts a simple yet effective approach by incorporating the loss function into both the instruction and response sequences. It seamlessly expands the training data, and regularizes the MLLMs from overly relying on language priors. Based on this merit, LIT achieves a significant relative improvement of up to 9% on comprehensive multimodal benchmarks, requiring no additional training data and incurring negligible computational overhead. Surprisingly, LIT attains exceptional fundamental visual capabilities, yielding up to an 18% improvement in captioning performance, while simultaneously alleviating hallucination in MLLMs. 

**Abstract (ZH)**: 我们提出LIT，这是一种视觉指令调优（VIT）的改进。虽然VIT为多模态大语言模型（MLLMs）提供了令人期待的多模态能力，但当前VIT的设计选择往往会导致过拟合和捷径学习，这可能会影响性能。这种差距源于过度重视指令遵循能力，而忽视了主动理解视觉信息。受到这一启发，LIT采用了一种简单而有效的方法，通过将损失函数同时纳入指令和响应序列中来扩展训练数据，并防止MLLMs过于依赖语言先验。凭借这一优势，LIT在全面的多模态基准测试上实现了高达9%的相对改进，无需额外的训练数据且几乎不增加计算开销。令人惊讶的是，LIT在基本视觉能力方面表现出色，captioning性能提高了高达18%，同时缓解了MLLMs中的幻觉现象。 

---
# Sell It Before You Make It: Revolutionizing E-Commerce with Personalized AI-Generated Items 

**Title (ZH)**: 在制造之前就卖掉它：以个性化AI生成物品革新电子商务 

**Authors**: Jianghao Lin, Peng Du, Jiaqi Liu, Weite Li, Yong Yu, Weinan Zhang, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22182)  

**Abstract**: E-commerce has revolutionized retail, yet its traditional workflows remain inefficient, with significant time and resource costs tied to product design and manufacturing inventory. This paper introduces a novel system deployed at Alibaba that leverages AI-generated items (AIGI) to address these challenges with personalized text-to-image generation for e-commercial product design. AIGI enables an innovative business mode called "sell it before you make it", where merchants can design fashion items and generate photorealistic images with digital models based on textual descriptions. Only when the items have received a certain number of orders, do the merchants start to produce them, which largely reduces reliance on physical prototypes and thus accelerates time to market. For such a promising application, we identify the underlying key scientific challenge, i.e., capturing the users' group-level personalized preferences towards multiple generated candidate images. To this end, we propose a Personalized Group-Level Preference Alignment Framework for Diffusion Models (i.e., PerFusion). We first design PerFusion Reward Model for user preference estimation with a feature-crossing-based personalized plug-in. Then we develop PerFusion with a personalized adaptive network to model diverse preferences across users, and meanwhile derive the group-level preference optimization objective to capture the comparative behaviors among multiple candidates. Both offline and online experiments demonstrate the effectiveness of our proposed algorithm. The AI-generated items have achieved over 13% relative improvements for both click-through rate and conversion rate compared to their human-designed counterparts, validating the revolutionary potential of AI-generated items for e-commercial platforms. 

**Abstract (ZH)**: 电商平台已经重塑了零售业，但其传统的 workflows仍然效率低下，产品设计和制造库存涉及大量时间和资源成本。本文介绍了一种部署在阿里巴巴的新系统，该系统利用AI生成的物品（AIGI）通过个性化文本到图像生成解决电商平台产品设计中的挑战，实现了“先卖后制”的创新商业模式。商家可以根据文本描述设计时尚商品并生成逼真的图像。只有当商品收到一定数量的订单后，商家才开始生产，这大大减少了对实物原型的依赖，从而加快了市场时间。针对这一有前景的应用，我们确定了其背后的科学挑战，即捕捉用户群体级个性化偏好对多个生成候选图像的理解。为此，我们提出了基于扩散模型的个性化群体级偏好对齐框架（即PerFusion）。我们首先设计了基于特征交叉的个性化插件的PerFusion奖励模型以进行用户偏好估计，然后开发了个性化自适应网络以建模用户的多样偏好，并同时推导出群体级偏好优化目标以捕捉多个候选者的相对行为。离线和在线实验均证明了我们提出算法的有效性。AI生成的物品在点击率和转换率上相对于人类设计的商品分别实现了超过13%的相对提升，验证了AI生成的商品在电商平台上具有革命性的潜力。 

---
# e-person Architecture and Framework for Human-AI Co-adventure Relationship 

**Title (ZH)**: 基于e-person架构的人机共venture关系框架 

**Authors**: Kanako Esaki, Tadayuki Matsumura, Yang Shao, Hiroyuki Mizuno  

**Link**: [PDF](https://arxiv.org/pdf/2503.22181)  

**Abstract**: This paper proposes the e-person architecture for constructing a unified and incremental development of AI ethics. The e-person architecture takes the reduction of uncertainty through collaborative cognition and action with others as a unified basis for ethics. By classifying and defining uncertainty along two axes - (1) first, second, and third person perspectives, and (2) the difficulty of inference based on the depth of information - we support the development of unified and incremental development of AI ethics. In addition, we propose the e-person framework based on the free energy principle, which considers the reduction of uncertainty as a unifying principle of brain function, with the aim of implementing the e-person architecture, and we show our previous works and future challenges based on the proposed framework. 

**Abstract (ZH)**: 本文提出e-person架构以构建统一和递增的人工智能伦理。e-person架构将通过与他人的协作认知和行动减少不确定性作为伦理的基础。通过沿着两个轴分类和定义不确定性——（1）第一人、第二人和第三人视角，（2）基于信息深度的推理难度——我们支持统一和递增的人工智能伦理的发展。此外，我们基于自由能量原则提出e-person框架，将减少不确定性视为大脑功能的统一原则，旨在实施e-person架构，并基于所提出框架展示我们的先前工作和未来挑战。 

---
# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging 

**Title (ZH)**: 自适应排名剪枝以增强模型合并 

**Authors**: Chanhyuk Lee, Jiho Choi, Chanryeol Lee, Donggyun Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22178)  

**Abstract**: Model merging has emerged as a promising approach for unifying independently fine-tuned models into an integrated framework, significantly enhancing computational efficiency in multi-task learning. Recently, several SVD-based techniques have been introduced to exploit low-rank structures for enhanced merging, but their reliance on such manually designed rank selection often leads to cross-task interference and suboptimal performance. In this paper, we propose AdaRank, a novel model merging framework that adaptively selects the most beneficial singular directions of task vectors to merge multiple models. We empirically show that the dominant singular components of task vectors can cause critical interference with other tasks, and that naive truncation across tasks and layers degrades performance. In contrast, AdaRank dynamically prunes the singular components that cause interference and offers an optimal amount of information to each task vector by learning to prune ranks during test-time via entropy minimization. Our analysis demonstrates that such method mitigates detrimental overlaps among tasks, while empirical results show that AdaRank consistently achieves state-of-the-art performance with various backbones and number of tasks, reducing the performance gap between fine-tuned models to nearly 1%. 

**Abstract (ZH)**: 自适应奇异值选择的模型融合框架：AdaRank 

---
# PharmAgents: Building a Virtual Pharma with Large Language Model Agents 

**Title (ZH)**: PharmAgents: 构建一个基于大规模语言模型代理的虚拟制药领域 

**Authors**: Bowen Gao, Yanwen Huang, Yiqiao Liu, Wenxuan Xie, Wei-Ying Ma, Ya-Qin Zhang, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22164)  

**Abstract**: The discovery of novel small molecule drugs remains a critical scientific challenge with far-reaching implications for treating diseases and advancing human health. Traditional drug development--especially for small molecule therapeutics--is a highly complex, resource-intensive, and time-consuming process that requires multidisciplinary collaboration. Recent breakthroughs in artificial intelligence (AI), particularly the rise of large language models (LLMs), present a transformative opportunity to streamline and accelerate this process. In this paper, we introduce PharmAgents, a virtual pharmaceutical ecosystem driven by LLM-based multi-agent collaboration. PharmAgents simulates the full drug discovery workflow--from target discovery to preclinical evaluation--by integrating explainable, LLM-driven agents equipped with specialized machine learning models and computational tools. Through structured knowledge exchange and automated optimization, PharmAgents identifies potential therapeutic targets, discovers promising lead compounds, enhances binding affinity and key molecular properties, and performs in silico analyses of toxicity and synthetic feasibility. Additionally, the system supports interpretability, agent interaction, and self-evolvement, enabling it to refine future drug designs based on prior experience. By showcasing the potential of LLM-powered multi-agent systems in drug discovery, this work establishes a new paradigm for autonomous, explainable, and scalable pharmaceutical research, with future extensions toward comprehensive drug lifecycle management. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统在药物发现中的应用： PharmAgents 虚拟制药生态系统 

---
# EgoToM: Benchmarking Theory of Mind Reasoning from Egocentric Videos 

**Title (ZH)**: EgoToM: 基于以自我为中心视频的情绪理论推理基准测试 

**Authors**: Yuxuan Li, Vijay Veerabadran, Michael L. Iuzzolino, Brett D. Roads, Asli Celikyilmaz, Karl Ridgeway  

**Link**: [PDF](https://arxiv.org/pdf/2503.22152)  

**Abstract**: We introduce EgoToM, a new video question-answering benchmark that extends Theory-of-Mind (ToM) evaluation to egocentric domains. Using a causal ToM model, we generate multi-choice video QA instances for the Ego4D dataset to benchmark the ability to predict a camera wearer's goals, beliefs, and next actions. We study the performance of both humans and state of the art multimodal large language models (MLLMs) on these three interconnected inference problems. Our evaluation shows that MLLMs achieve close to human-level accuracy on inferring goals from egocentric videos. However, MLLMs (including the largest ones we tested with over 100B parameters) fall short of human performance when inferring the camera wearers' in-the-moment belief states and future actions that are most consistent with the unseen video future. We believe that our results will shape the future design of an important class of egocentric digital assistants which are equipped with a reasonable model of the user's internal mental states. 

**Abstract (ZH)**: EgoToM：扩展到主体中心领域的理论-of-心智视频问答基准 

---
# When Autonomy Breaks: The Hidden Existential Risk of AI 

**Title (ZH)**: 当自主性失效：AI潜藏的 existential 风险 

**Authors**: Joshua Krook  

**Link**: [PDF](https://arxiv.org/pdf/2503.22151)  

**Abstract**: AI risks are typically framed around physical threats to humanity, a loss of control or an accidental error causing humanity's extinction. However, I argue in line with the gradual disempowerment thesis, that there is an underappreciated risk in the slow and irrevocable decline of human autonomy. As AI starts to outcompete humans in various areas of life, a tipping point will be reached where it no longer makes sense to rely on human decision-making, creativity, social care or even leadership.
What may follow is a process of gradual de-skilling, where we lose skills that we currently take for granted. Traditionally, it is argued that AI will gain human skills over time, and that these skills are innate and immutable in humans. By contrast, I argue that humans may lose such skills as critical thinking, decision-making and even social care in an AGI world. The biggest threat to humanity is therefore not that machines will become more like humans, but that humans will become more like machines. 

**Abstract (ZH)**: AI风险通常围绕着对人类的物理威胁、控制丧失或偶然错误导致人类灭绝。然而，我沿着逐步失能理论的立场argue，人类自主性的缓慢而不可逆转的下降被低估了这一风险。随着AI在生活各个领域的竞争力超越人类，将达到一个转折点，在这一点上，依赖人类决策、创造力、社会关怀或甚至领导能力就不再有意义了。随之而来的可能是技能逐步退化的过程，我们可能会失去现在习以为常的技能。传统观点认为，AI将随着时间获得人类技能，而这些技能在人类身上是天生不可变的。相反，我认为在通用人工智能的世界中，人类可能会失去批判性思维、决策能力甚至社会关怀等技能。因此，人类面临的最大威胁不是机器会变得越来越像人，而是人会变得越来越像机器。 

---
# FRASE: Structured Representations for Generalizable SPARQL Query Generation 

**Title (ZH)**: FRASE: 结构化表示以生成可泛化的SPARQL查询 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2503.22144)  

**Abstract**: Translating natural language questions into SPARQL queries enables Knowledge Base querying for factual and up-to-date responses. However, existing datasets for this task are predominantly template-based, leading models to learn superficial mappings between question and query templates rather than developing true generalization capabilities. As a result, models struggle when encountering naturally phrased, template-free questions. This paper introduces FRASE (FRAme-based Semantic Enhancement), a novel approach that leverages Frame Semantic Role Labeling (FSRL) to address this limitation. We also present LC-QuAD 3.0, a new dataset derived from LC-QuAD 2.0, in which each question is enriched using FRASE through frame detection and the mapping of frame-elements to their argument. We evaluate the impact of this approach through extensive experiments on recent large language models (LLMs) under different fine-tuning configurations. Our results demonstrate that integrating frame-based structured representations consistently improves SPARQL generation performance, particularly in challenging generalization scenarios when test questions feature unseen templates (unknown template splits) and when they are all naturally phrased (reformulated questions). 

**Abstract (ZH)**: 基于框架语义增强的自然语言问题到SPARQL查询的转换：FRASE方法及其在LC-QuAD 3.0数据集上的应用 

---
# A Self-Supervised Learning of a Foundation Model for Analog Layout Design Automation 

**Title (ZH)**: 自监督学习为基础模型的模拟布局自动化设计 

**Authors**: Sungyu Jeong, Won Joon Choi, Junung Choi, Anik Biswas, Byungsub Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.22143)  

**Abstract**: We propose a UNet-based foundation model and its self-supervised learning method to address two key challenges: 1) lack of qualified annotated analog layout data, and 2) excessive variety in analog layout design tasks. For self-supervised learning, we propose random patch sampling and random masking techniques automatically to obtain enough training data from a small unannotated layout dataset. The obtained data are greatly augmented, less biased, equally sized, and contain enough information for excessive varieties of qualified layout patterns. By pre-training with the obtained data, the proposed foundation model can learn implicit general knowledge on layout patterns so that it can be fine-tuned for various downstream layout tasks with small task-specific datasets. Fine-tuning provides an efficient and consolidated methodology for diverse downstream tasks, reducing the enormous human effort to develop a model per task separately. In experiments, the foundation model was pre-trained using 324,000 samples obtained from 6 silicon-proved manually designed analog circuits, then it was fine-tuned for the five example downstream tasks: generating contacts, vias, dummy fingers, N-wells, and metal routings. The fine-tuned models successfully performed these tasks for more than one thousand unseen layout inputs, generating DRC/LVS-clean layouts for 96.6% of samples. Compared with training the model from scratch for the metal routing task, fine-tuning required only 1/8 of the data to achieve the same dice score of 0.95. With the same data, fine-tuning achieved a 90% lower validation loss and a 40% higher benchmark score than training from scratch. 

**Abstract (ZH)**: 基于UNet的基础模型及其自监督学习方法以应对模拟版图设计中的两个关键挑战 

---
# Integrating Artificial Intelligence with Human Expertise: An In-depth Analysis of ChatGPT's Capabilities in Generating Metamorphic Relations 

**Title (ZH)**: 将人工智能与人类 expertise 结合：ChatGPT 在生成元变关系方面的能力深入分析 

**Authors**: Yifan Zhang, Dave Towey, Matthew Pike, Quang-Hung Luu, Huai Liu, Tsong Yueh Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22141)  

**Abstract**: Context: This paper provides an in-depth examination of the generation and evaluation of Metamorphic Relations (MRs) using GPT models developed by OpenAI, with a particular focus on the capabilities of GPT-4 in software testing environments.
Objective: The aim is to examine the quality of MRs produced by GPT-3.5 and GPT-4 for a specific System Under Test (SUT) adopted from an earlier study, and to introduce and apply an improved set of evaluation criteria for a diverse range of SUTs.
Method: The initial phase evaluates MRs generated by GPT-3.5 and GPT-4 using criteria from a prior study, followed by an application of an enhanced evaluation framework on MRs created by GPT-4 for a diverse range of nine SUTs, varying from simple programs to complex systems incorporating AI/ML components. A custom-built GPT evaluator, alongside human evaluators, assessed the MRs, enabling a direct comparison between automated and human evaluation methods.
Results: The study finds that GPT-4 outperforms GPT-3.5 in generating accurate and useful MRs. With the advanced evaluation criteria, GPT-4 demonstrates a significant ability to produce high-quality MRs across a wide range of SUTs, including complex systems incorporating AI/ML components.
Conclusions: GPT-4 exhibits advanced capabilities in generating MRs suitable for various applications. The research underscores the growing potential of AI in software testing, particularly in the generation and evaluation of MRs, and points towards the complementarity of human and AI skills in this domain. 

**Abstract (ZH)**: 基于GPT模型的元变关系生成与评估：以OpenAI GPT-4在软件测试环境中的能力为重点 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Title (ZH)**: REMAC: 具有自我反思和自我进化的多代理协作机器人长时 horizon 操控 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

**Abstract (ZH)**: 视觉-语言模型在机器人规划中的应用：面向长时程任务的自适应多 Agent 计划框架 REMAC 

---
# Beyond Single-Sentence Prompts: Upgrading Value Alignment Benchmarks with Dialogues and Stories 

**Title (ZH)**: 超越单句提示：通过对话和故事提升价值对齐基准 

**Authors**: Yazhou Zhang, Qimeng Liu, Qiuchi Li, Peng Zhang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22115)  

**Abstract**: Evaluating the value alignment of large language models (LLMs) has traditionally relied on single-sentence adversarial prompts, which directly probe models with ethically sensitive or controversial questions. However, with the rapid advancements in AI safety techniques, models have become increasingly adept at circumventing these straightforward tests, limiting their effectiveness in revealing underlying biases and ethical stances. To address this limitation, we propose an upgraded value alignment benchmark that moves beyond single-sentence prompts by incorporating multi-turn dialogues and narrative-based scenarios. This approach enhances the stealth and adversarial nature of the evaluation, making it more robust against superficial safeguards implemented in modern LLMs. We design and implement a dataset that includes conversational traps and ethically ambiguous storytelling, systematically assessing LLMs' responses in more nuanced and context-rich settings. Experimental results demonstrate that this enhanced methodology can effectively expose latent biases that remain undetected in traditional single-shot evaluations. Our findings highlight the necessity of contextual and dynamic testing for value alignment in LLMs, paving the way for more sophisticated and realistic assessments of AI ethics and safety. 

**Abstract (ZH)**: 评估大型语言模型的价值对齐 traditionally 依赖于单句对抗提示，这些提示直接用伦理敏感或有争议的问题来测试模型。然而，随着人工智能安全技术的迅速发展，模型越来越擅长绕过这些简单的测试，限制了它们在揭示潜在偏差和伦理立场方面的有效性。为了弥补这一不足，我们提出了一种升级的价值对齐基准，超越了单句提示，通过加入多轮对话和叙述性场景来增强评估的隐蔽性和对抗性，使其更能抵抗现代大型语言模型中实施的表面性保护措施。我们设计并实现了一个包含对话陷阱和伦理含糊故事的數據集，系统评估了大型语言模型在更为细腻和情境丰富的设置下对这些场景的回应。实验结果表明，这一改进的方法能够有效揭示传统单一测试中未检测到的潜在偏差。我们的研究突显了在大型语言模型中进行价值对齐测试时的必要性，即需要进行上下文性和动态性测试，为更复杂和现实的AI伦理和安全评估奠定了基础。 

---
# How Well Can Vison-Language Models Understand Humans' Intention? An Open-ended Theory of Mind Question Evaluation Benchmark 

**Title (ZH)**: 视觉-语言模型能多准确地理解人类的意图？一个开放性理论思维问题评价基准 

**Authors**: Ximing Wen, Mallika Mainali, Anik Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22093)  

**Abstract**: Vision Language Models (VLMs) have demonstrated strong reasoning capabilities in Visual Question Answering (VQA) tasks; However, their ability to perform Theory of Mind (ToM) tasks such as accurately inferring human intentions, beliefs, and other mental states remains underexplored. In this work, we propose an open-ended question framework to comprehensively evaluate VLMs' performance across diverse categories of ToM tasks. We curated and annotated a benchmark dataset composed of 30 images. We then assessed the performance of four VLMs of varying sizes on this dataset. Our experimental results show that the GPT-4 model outperformed all others, with only one smaller model, GPT-4o-mini, achieving comparable performance. Additionally, we observed that VLMs often struggle to accurately infer intentions in complex scenarios such as bullying or cheating. Moreover, our findings also reveal that smaller models can sometimes infer correct intentions despite relying on incorrect visual cues. 

**Abstract (ZH)**: 视觉语言模型在同理心任务中的表现：一个开放性问题框架的探索 

---
# Penrose Tiled Low-Rank Compression and Section-Wise Q&A Fine-Tuning: A General Framework for Domain-Specific Large Language Model Adaptation 

**Title (ZH)**: 佩恩罗斯铺砖低秩压缩与段落wise问答微调：一种领域特定大型语言模型适应的一般框架 

**Authors**: Chuan-Wei Kuo, Siyu Chen, Chenqi Yan, Yu Yang Fredrik Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22074)  

**Abstract**: Large language models (LLMs) hold great promise for specialized scientific domains such as materials science, yet adapting them efficiently and accurately to domain-specific knowledge remains challenging due to limited data and high knowledge density. We propose a two-stage framework that combines structured model compression with a scientific fine-tuning regimen to address this challenge. In the compression stage, we decompose the LLM's weight matrices into local low-rank "rank blocks" and arrange these blocks in a Penrose-like non-periodic tiling pattern. Each block is then compacted via spectral transformations (e.g., discrete cosine or Fourier transforms), and a Kullback-Leibler (KL) divergence-based alignment loss preserves the distributional similarity between the compressed model's representations and those of the original full model. In the adaptation stage, the compressed model is further tuned using a human-like scientific reading protocol: it processes technical materials science documents section by section, engaging in a structured question-and-answer routine for each section. This section-wise Q&A fine-tuning strategy extracts explicit reasoning traces and gradually injects domain knowledge, while minimizing catastrophic forgetting of the model's general language capabilities. By balancing efficient compression with targeted adaptation, our two-stage approach enables precise specialization of LLMs to high-value domains under data-scarce conditions. We present this principled yet exploratory pipeline and outline its potential for advancing materials science knowledge integration, laying the groundwork for comprehensive empirical evaluation in future work. 

**Abstract (ZH)**: 大型语言模型（LLMs）在材料科学等专业科学领域展现出巨大的潜力，但由于数据有限和知识密度高，高效且准确地将它们适应到特定领域的知识仍然颇具挑战。我们提出了一种两阶段框架，结合结构化模型压缩与科学调优方案，以应对这一挑战。在压缩阶段，我们将LLM的权重矩阵分解为局部低秩“秩块”，并通过Penrose-like非周期镶嵌模式排列这些块。随后，通过谱变换（如离散余弦变换或傅里叶变换）对每个块进行压缩，并通过基于Kullback-Leibler散度的对齐损失保持压缩模型表示与原始完整模型表示之间的分布相似性。在适应阶段，使用类似人类的科学阅读协议对压缩模型进行进一步调优：逐节处理技术材料科学文档，并针对每个部分进行结构化的问答惯例。这种按节问答的调优策略提取显式的推理轨迹，并逐渐注入领域知识，同时尽可能减少模型对一般语言能力的灾难性遗忘。通过在高效压缩与目标化适应之间取得平衡，我们的两阶段方法在数据稀缺条件下使LLM能够实现精准的专业化。我们将这一原理性的探索管线呈现出来，并概述其在推进材料科学知识整合方面的潜在价值，为未来进行全面实证评估奠定基础。 

---
# Contrasting Low and High-Resolution Features for HER2 Scoring using Deep Learning 

**Title (ZH)**: 低分辨率与高分辨率特征在使用深度学习进行HER2评分中的对比 

**Authors**: Ekansh Chauhan, Anila Sharma, Amit Sharma, Vikas Nishadham, Asha Ghughtyal, Ankur Kumar, Gurudutt Gupta, Anurag Mehta, C.V. Jawahar, P.K. Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2503.22069)  

**Abstract**: Breast cancer, the most common malignancy among women, requires precise detection and classification for effective treatment. Immunohistochemistry (IHC) biomarkers like HER2, ER, and PR are critical for identifying breast cancer subtypes. However, traditional IHC classification relies on pathologists' expertise, making it labor-intensive and subject to significant inter-observer variability. To address these challenges, this study introduces the India Pathology Breast Cancer Dataset (IPD-Breast), comprising of 1,272 IHC slides (HER2, ER, and PR) aimed at automating receptor status classification. The primary focus is on developing predictive models for HER2 3-way classification (0, Low, High) to enhance prognosis. Evaluation of multiple deep learning models revealed that an end-to-end ConvNeXt network utilizing low-resolution IHC images achieved an AUC, F1, and accuracy of 91.79%, 83.52%, and 83.56%, respectively, for 3-way classification, outperforming patch-based methods by over 5.35% in F1 score. This study highlights the potential of simple yet effective deep learning techniques to significantly improve accuracy and reproducibility in breast cancer classification, supporting their integration into clinical workflows for better patient outcomes. 

**Abstract (ZH)**: 乳腺癌，女性最常见的恶性肿瘤，要求精确检测和分类以实现有效的治疗。免疫组织化学（IHC）标志物如HER2、ER和PR对于识别乳腺癌亚型至关重要。然而，传统的IHC分类依赖于病理学家的专业知识，这使得其劳动密集且具有显著的观察者间变异。为了应对这些挑战，本研究引入了印度病理学乳腺癌数据集（IPD-Breast），包含1,272张IHC切片（HER2、ER和PR），旨在实现受体状态的自动化分类。主要重点是开发HER2三分类（0、低、高）的预测模型，以提高预后。多种深度学习模型的评估表明，使用低分辨率IHC图像的端到端ConvNeXt网络在三分类中的AUC、F1和准确率分别为91.79%、83.52%和83.56%，其F1得分比基于patch的方法高出5.35%以上。本研究突显了简单有效的深度学习技术在提高乳腺癌分类的准确性和可重复性方面的潜力，支持其集成到临床工作流程中以改善患者预后。 

---
# A Proposal for Networks Capable of Continual Learning 

**Title (ZH)**: 持续学习能力网络的提案 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2503.22068)  

**Abstract**: We analyze the ability of computational units to retain past responses after parameter updates, a key property for system-wide continual learning. Neural networks trained with gradient descent lack this capability, prompting us to propose Modelleyen, an alternative approach with inherent response preservation. We demonstrate through experiments on modeling the dynamics of a simple environment and on MNIST that, despite increased computational complexity and some representational limitations at its current stage, Modelleyen achieves continual learning without relying on sample replay or predefined task boundaries. 

**Abstract (ZH)**: 我们分析了计算单元在参数更新后保留过去响应的能力，这是系统级连续学习的关键属性。尽管Modelleyen目前存在一些计算复杂度增加和表示限制，但我们的实验表明，它能够在不依赖样本重放或预定义任务边界的情况下实现连续学习。 

---
# Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation 

**Title (ZH)**: 基于非单调注意力的读写策略学习以实现同步翻译 

**Authors**: Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin, Jachym Kolar, Niko Moritz, Ruiming Xie, Simone Merello, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22051)  

**Abstract**: Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model. 

**Abstract (ZH)**: 同时或流式机器翻译在阅读输入流的同时生成翻译。这些系统面临质量和延迟之间的权衡，旨在最小延迟的情况下实现类似非流式模型的高翻译质量。我们提出了一种有效管理这种权衡的方法。通过增强一个用 seq2seq 机制预训练且代表质量上限的非流式模型，利用源和目标词之间的对齐，将其转换为流式模型。利用这种对齐来学习一个读写决策边界，以实现可靠的最小输入翻译生成。在训练过程中，模型通过读写策略模块学习决策边界，该模块在对齐点（伪标签）上使用有监督学习。读写策略模块作为一个小型二元分类单元，在推断过程中可以控制质量和延迟之间的权衡。实验结果表明，我们的模型优于多个强基准模型，并且与非流式基准模型之间的差距缩小。 

---
# Cognitive Prompts Using Guilford's Structure of Intellect Model 

**Title (ZH)**: 基于吉尔福特结构智力模型的认知提示方法 

**Authors**: Oliver Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2503.22036)  

**Abstract**: Large language models (LLMs) demonstrate strong language generation capabilities but often struggle with structured reasoning, leading to inconsistent or suboptimal problem-solving. To mitigate this limitation, Guilford's Structure of Intellect (SOI) model - a foundational framework from intelligence theory - is leveraged as the basis for cognitive prompt engineering. The SOI model categorizes cognitive operations such as pattern recognition, memory retrieval, and evaluation, offering a systematic approach to enhancing LLM reasoning and decision-making. This position paper presents a novel cognitive prompting approach for enforcing SOI-inspired reasoning for improving clarity, coherence, and adaptability in model responses. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了强大的语言生成能力，但在结构化推理方面往往表现不佳，导致问题解决的一致性或最优性不足。为了缓解这一限制，本文以智力理论中的Guilford的结构要素（SOI）模型为基础，提出了一种认知提示工程方法。SOI模型将认知操作分为模式识别、记忆检索和评估等类别，提供了一种系统的方法来提升LLM的推理和决策能力。本文提出了一种新颖的认知提示方法，旨在通过SOI启发式的推理提高模型响应的清晰度、连贯性和适应性。 

---
# Safeguarding Autonomy: a Focus on Machine Learning Decision Systems 

**Title (ZH)**: 保障自主权：关注机器学习决策系统 

**Authors**: Paula Subías-Beltrán, Oriol Pujol, Itziar de Lecuona  

**Link**: [PDF](https://arxiv.org/pdf/2503.22023)  

**Abstract**: As global discourse on AI regulation gains momentum, this paper focuses on delineating the impact of ML on autonomy and fostering awareness. Respect for autonomy is a basic principle in bioethics that establishes persons as decision-makers. While the concept of autonomy in the context of ML appears in several European normative publications, it remains a theoretical concept that has yet to be widely accepted in ML practice. Our contribution is to bridge the theoretical and practical gap by encouraging the practical application of autonomy in decision-making within ML practice by identifying the conditioning factors that currently prevent it. Consequently, we focus on the different stages of the ML pipeline to identify the potential effects on ML end-users' autonomy. To improve its practical utility, we propose a related question for each detected impact, offering guidance for identifying possible focus points to respect ML end-users autonomy in decision-making. 

**Abstract (ZH)**: 随着全球对AI监管的讨论不断升温，本文旨在阐明ML对自主性的影响，并提高对此类问题的认识。在生物伦理学中，尊重自主性是一个基本原则，确立了个人作为决策者的地位。虽然在ML背景下自主性的概念已在多个欧洲规范性出版物中出现，但这一概念在ML实践中尚未得到广泛接受。我们的贡献是通过识别目前阻碍其应用的因素，来弥合理论与实践之间的差距，鼓励在ML实践中将自主性应用于决策过程。因此，我们重点关注ML管道的不同阶段，以识别对ML终端用户自主性潜在的影响。为了提高其实用性，我们针对每个检测到的影响提出相关问题，为识别可能的重点以尊重ML终端用户在决策中的自主性提供指导。 

---
# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models 

**Title (ZH)**: CoT-VLA: 视觉链式思考推理方法在视语动模型中的应用 

**Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22020)  

**Abstract**: Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLAs primarily focus on direct input--output mappings, lacking the intermediate reasoning steps crucial for complex manipulation tasks. As a result, existing VLAs lack temporal planning or reasoning capabilities. In this paper, we introduce a method that incorporates explicit visual chain-of-thought (CoT) reasoning into vision-language-action models (VLAs) by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. We introduce CoT-VLA, a state-of-the-art 7B VLA that can understand and generate visual and action tokens. Our experimental results demonstrate that CoT-VLA achieves strong performance, outperforming the state-of-the-art VLA model by 17% in real-world manipulation tasks and 6% in simulation benchmarks. Project website: this https URL 

**Abstract (ZH)**: 基于视觉-语言-动作模型（VLAs）的显式视觉链式思考推理方法 

---
# BOOTPLACE: Bootstrapped Object Placement with Detection Transformers 

**Title (ZH)**: BOOTPLACE: 基于检测变换器的自举对象放置 

**Authors**: Hang Zhou, Xinxin Zuo, Rui Ma, Li Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21991)  

**Abstract**: In this paper, we tackle the copy-paste image-to-image composition problem with a focus on object placement learning. Prior methods have leveraged generative models to reduce the reliance for dense supervision. However, this often limits their capacity to model complex data distributions. Alternatively, transformer networks with a sparse contrastive loss have been explored, but their over-relaxed regularization often leads to imprecise object placement. We introduce BOOTPLACE, a novel paradigm that formulates object placement as a placement-by-detection problem. Our approach begins by identifying suitable regions of interest for object placement. This is achieved by training a specialized detection transformer on object-subtracted backgrounds, enhanced with multi-object supervisions. It then semantically associates each target compositing object with detected regions based on their complementary characteristics. Through a boostrapped training approach applied to randomly object-subtracted images, our model enforces meaningful placements through extensive paired data augmentation. Experimental results on established benchmarks demonstrate BOOTPLACE's superior performance in object repositioning, markedly surpassing state-of-the-art baselines on Cityscapes and OPA datasets with notable improvements in IOU scores. Additional ablation studies further showcase the compositionality and generalizability of our approach, supported by user study evaluations. 

**Abstract (ZH)**: 本文介绍了BOOTPLACE，一种新颖的物体放置学习范式，将物体放置问题形式化为检测问题。通过在物体减背景上训练专门的检测变压器，并结合多物体监督，我们的方法首先识别适合放置物体的感兴趣区域。然后基于检测到的区域和目标组成物体的互补特征进行语义关联。通过应用于随机物体减背景图像的自助训练方法，我们的模型通过广泛的配对数据增强强制实现有意义的放置。在建立的基线上的实验结果表明，BOOTPLACE在物体重新定位任务中的性能显著优于现有最先进的基线，特别是在Cityscapes和OPA数据集上，IOU分数有显著提高。额外的消融研究进一步展示了我们方法的组合性和泛化能力，并得到了用户研究的验证。 

---
# Pretrained Bayesian Non-parametric Knowledge Prior in Robotic Long-Horizon Reinforcement Learning 

**Title (ZH)**: 预训练贝叶斯非参数先验知识在机器人长 horizon 强化学习中的应用 

**Authors**: Yuan Meng, Xiangtong Yao, Kejia Chen, Yansong Wu, Liding Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21975)  

**Abstract**: Reinforcement learning (RL) methods typically learn new tasks from scratch, often disregarding prior knowledge that could accelerate the learning process. While some methods incorporate previously learned skills, they usually rely on a fixed structure, such as a single Gaussian distribution, to define skill priors. This rigid assumption can restrict the diversity and flexibility of skills, particularly in complex, long-horizon tasks. In this work, we introduce a method that models potential primitive skill motions as having non-parametric properties with an unknown number of underlying features. We utilize a Bayesian non-parametric model, specifically Dirichlet Process Mixtures, enhanced with birth and merge heuristics, to pre-train a skill prior that effectively captures the diverse nature of skills. Additionally, the learned skills are explicitly trackable within the prior space, enhancing interpretability and control. By integrating this flexible skill prior into an RL framework, our approach surpasses existing methods in long-horizon manipulation tasks, enabling more efficient skill transfer and task success in complex environments. Our findings show that a richer, non-parametric representation of skill priors significantly improves both the learning and execution of challenging robotic tasks. All data, code, and videos are available at this https URL. 

**Abstract (ZH)**: 强化学习方法通常从头学习新任务，往往忽视可以加速学习过程的先验知识。虽然有些方法整合了先前学习的技能，但它们通常依赖于固定的结构，如单一的高斯分布，来定义技能先验。这种刚性假设可能会限制技能的多样性和灵活性，特别是在复杂的长期任务中。在本文中，我们提出了一种方法，将其潜在的基本技能运动模型化为具有非参数特性的未知数量的底层特征。我们利用Bayesian非参数模型，特别是Dirichlet过程混合模型，并结合出生和合并启发式方法，预训练一个能有效捕捉技能多样性的技能先验。此外，学习到的技能在先验空间中是显式可追踪的，增强了可解释性和控制性。通过将这种灵活的技能先验整合到RL框架中，我们的方法在长期操作任务上超越了现有方法，促进了更高效的技术掌握和复杂环境中的任务成功。我们的研究结果表明，更丰富、非参数化的技能先验表示显著地提高了挑战性机器人任务的学习和执行效果。所有数据、代码和视频都可以在以下网址获取。 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Title (ZH)**: 基于视觉-语言引导的闭环反馈的无数据驱动长_horizon机械臂操作 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

**Abstract (ZH)**: 语言条件下的机器人操作 recent 进展：DAHLIA——一种基于大规模语言模型的数据无关框架，用于长时 Horizon 任务执行 

---
# Entropy-Aware Branching for Improved Mathematical Reasoning 

**Title (ZH)**: 熵意识分支以提高数学推理能力 

**Authors**: Xianzhi Li, Ethan Callanan, Xiaodan Zhu, Mathieu Sibue, Antony Papadimitriou, Mahmoud Mahfouz, Zhiqiang Ma, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21961)  

**Abstract**: While Large Language Models (LLMs) are effectively aligned through extensive pre-training and fine-tuning, they still struggle with varying levels of uncertainty during token generation. In our investigation of mathematical reasoning, we observe that errors are more likely to arise at tokens exhibiting high entropy and variance of entropy in the model's output distribution. Based on the observation, we propose a novel approach that dynamically branches the generation process on demand instead of defaulting to the single most probable token. By exploring in parallel multiple branches stemming from high probability tokens of critical decision points, the model can discover diverse reasoning paths that might otherwise be missed. We further harness external feedback from larger models to rank and select the most coherent and accurate reasoning branch. Our experimental results on mathematical word problems and calculation questions show that this branching strategy boosts the reasoning capabilities of small LLMs up to 4.6% compared to conventional argmax decoding. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）通过广泛的预训练和微调实现了有效的对齐，但在标记生成过程中仍然面临着不同程度的不确定性挑战。在我们对数学推理的研究中，我们观察到高熵及其变化的令牌更可能产生错误。基于这一观察，我们提出了一种新的方法，该方法在需求时动态分支生成过程，而不是默认选择最可能的令牌。通过并行探索关键决策点高概率令牌分支，模型可以发现可能被遗漏的多样化推理路径。我们进一步利用更大模型的外部反馈来评估和选择最连贯和准确的推理分支。实验结果表明，与传统的argmax解码方法相比，这种分支策略可以将小型LLM的推理能力提高4.6%。 

---
# Parametric Shadow Control for Portrait Generationin Text-to-Image Diffusion Models 

**Title (ZH)**: 面向肖像生成的参数化阴影控制在文本到图像扩散模型中 

**Authors**: Haoming Cai, Tsung-Wei Huang, Shiv Gehlot, Brandon Y. Feng, Sachin Shah, Guan-Ming Su, Christopher Metzler  

**Link**: [PDF](https://arxiv.org/pdf/2503.21943)  

**Abstract**: Text-to-image diffusion models excel at generating diverse portraits, but lack intuitive shadow control. Existing editing approaches, as post-processing, struggle to offer effective manipulation across diverse styles. Additionally, these methods either rely on expensive real-world light-stage data collection or require extensive computational resources for training. To address these limitations, we introduce Shadow Director, a method that extracts and manipulates hidden shadow attributes within well-trained diffusion models. Our approach uses a small estimation network that requires only a few thousand synthetic images and hours of training-no costly real-world light-stage data needed. Shadow Director enables parametric and intuitive control over shadow shape, placement, and intensity during portrait generation while preserving artistic integrity and identity across diverse styles. Despite training only on synthetic data built on real-world identities, it generalizes effectively to generated portraits with diverse styles, making it a more accessible and resource-friendly solution. 

**Abstract (ZH)**: 基于文本描述的图像扩散模型在生成多样化的肖像方面表现出色，但缺乏直观的阴影控制。现有的编辑方法作为后处理手段，在不同风格下难以实现有效的操作。此外，这些方法要么依赖昂贵的现实世界光源采集数据，要么需要大量的计算资源进行训练。为解决这些局限性，我们引入了Shadow Director方法，该方法从well-trained扩散模型中提取和操控隐藏的阴影属性。我们的方法使用一个小的估计网络，仅需少量（几千张）合成图像和几小时的训练时间，无需昂贵的现实世界光源采集数据。Shadow Director在肖像生成过程中提供了参数化和直观的阴影形状、位置和强度控制，同时保持了不同风格下的艺术完整性和身份特征。尽管仅在基于真实身份的合成数据上进行训练，但其能够有效泛化到具有不同风格的生成肖像，使其更具可访问性和资源友好性。 

---
# Lobster: A GPU-Accelerated Framework for Neurosymbolic Programming 

**Title (ZH)**: lobster：一种基于GPU加速的神经符号编程框架 

**Authors**: Paul Biberstein, Ziyang Li, Joseph Devietti, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2503.21937)  

**Abstract**: Neurosymbolic programs combine deep learning with symbolic reasoning to achieve better data efficiency, interpretability, and generalizability compared to standalone deep learning approaches. However, existing neurosymbolic learning frameworks implement an uneasy marriage between a highly scalable, GPU-accelerated neural component with a slower symbolic component that runs on CPUs. We propose Lobster, a unified framework for harnessing GPUs in an end-to-end manner for neurosymbolic learning. Lobster maps a general neurosymbolic language based on Datalog to the GPU programming paradigm. This mapping is implemented via compilation to a new intermediate language called APM. The extra abstraction provided by APM allows Lobster to be both flexible, supporting discrete, probabilistic, and differentiable modes of reasoning on GPU hardware with a library of provenance semirings, and performant, implementing new optimization passes. We demonstrate that Lobster programs can solve interesting problems spanning the domains of natural language processing, image processing, program reasoning, bioinformatics, and planning. On a suite of 8 applications, Lobster achieves an average speedup of 5.3x over Scallop, a state-of-the-art neurosymbolic framework, and enables scaling of neurosymbolic solutions to previously infeasible tasks. 

**Abstract (ZH)**: 神经符号程序结合深度学习与符号推理，以实现与独立深度学习方法相比更好的数据效率、可解释性和泛化能力。然而，现有的神经符号学习框架在高度可扩展的GPU加速神经组件与在CPU上运行的较慢的符号组件之间实现了不协调的结合。我们提出Lobster，一种端到端利用GPU的统一神经符号学习框架。Lobster将基于Datalog的通用神经符号语言映射到GPU编程 paradigm。这种映射通过编译到一种新的中间语言APM来实现。APM提供的额外抽象使Lobster能够灵活地支持在GPU硬件上进行离散的、概率的和可微分的推理，并且高效地实现新的优化遍历。我们展示了Lobster程序可以解决涵盖自然语言处理、图像处理、程序推理、生物信息学和规划等领域的问题。在一系列8个应用程序上，Lobster的平均加速比最先进的神经符号框架Scallop快5.3倍，并使神经符号解决方案能够缩放到此前不可行的任务。 

---
# An Efficient Training Algorithm for Models with Block-wise Sparsity 

**Title (ZH)**: 块状稀疏模型的高效训练算法 

**Authors**: Ding Zhu, Zhiqun Zuo, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.21928)  

**Abstract**: Large-scale machine learning (ML) models are increasingly being used in critical domains like education, lending, recruitment, healthcare, criminal justice, etc. However, the training, deployment, and utilization of these models demand substantial computational resources. To decrease computation and memory costs, machine learning models with sparse weight matrices are widely used in the literature. Among sparse models, those with special sparse structures (e.g., models with block-wise sparse weight matrices) fit better with the hardware accelerators and can decrease the memory and computation costs during the inference. Unfortunately, while there are several efficient training methods, none of them are designed to train a block-wise sparse model efficiently. As a result, the current methods for training block-wise sparse models start with full and dense models leading to inefficient training. In this work, we focus on training models with \textit{block-wise sparse matrices} and propose an efficient training algorithm to decrease both computation and memory costs during training and inference. In addition, we will show that our proposed method enables us to efficiently find the right block size for the sparsity pattern during the training process. Our extensive empirical and theoretical analyses show that our algorithms can decrease the computation and memory costs significantly without a performance drop compared to baselines. 

**Abstract (ZH)**: 大规模机器学习模型在教育、信贷、招聘、医疗保健、刑事司法等领域被日益广泛地应用。然而，这些模型的训练、部署和利用需要大量的计算资源。为了降低计算和内存成本，文献中广泛使用了稀疏权重矩阵的机器学习模型。在稀疏模型中，具有特殊稀疏结构的模型（例如，具有块状稀疏权重矩阵的模型）更适合硬件加速器，并能在推理过程中减少内存和计算成本。不幸的是，尽管存在多种高效的训练方法，但它们都不是为高效训练块状稀疏模型设计的。因此，当前训练块状稀疏模型的方法从全稠密模型开始，导致训练效率低下。在本文中，我们专注于训练具有块状稀疏矩阵的模型，并提出了一种高效训练算法，以在训练和推理过程中降低计算和内存成本。此外，我们将展示我们的方法使我们能够在训练过程中高效地找到合适的块大小。广泛的实验和理论分析表明，我们的算法与基线方法相比，在性能不下降的情况下，可以显著减少计算和内存成本。 

---
# AutoPsyC: Automatic Recognition of Psychodynamic Conflicts from Semi-structured Interviews with Large Language Models 

**Title (ZH)**: AutoPsyC：从半结构化访谈中自动识别动力冲突的大语言模型方法 

**Authors**: Sayed Muddashir Hossain, Simon Ostermann, Patrick Gebhard, Cord Benecke, Josef van Genabith, Philipp Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.21911)  

**Abstract**: Psychodynamic conflicts are persistent, often unconscious themes that shape a person's behaviour and experiences. Accurate diagnosis of psychodynamic conflicts is crucial for effective patient treatment and is commonly done via long, manually scored semi-structured interviews. Existing automated solutions for psychiatric diagnosis tend to focus on the recognition of broad disorder categories such as depression, and it is unclear to what extent psychodynamic conflicts which even the patient themselves may not have conscious access to could be automatically recognised from conversation. In this paper, we propose AutoPsyC, the first method for recognising the presence and significance of psychodynamic conflicts from full-length Operationalized Psychodynamic Diagnostics (OPD) interviews using Large Language Models (LLMs). Our approach combines recent advances in parameter-efficient fine-tuning and Retrieval-Augmented Generation (RAG) with a summarisation strategy to effectively process entire 90 minute long conversations. In evaluations on a dataset of 141 diagnostic interviews we show that AutoPsyC consistently outperforms all baselines and ablation conditions on the recognition of four highly relevant psychodynamic conflicts. 

**Abstract (ZH)**: 动心理冲突是持久的、Often未意识的模式，影响人的行为和体验。准确诊断动心理冲突对于有效的患者治疗至关重要，通常通过长时间的手动评分半结构化访谈来完成。现有的精神疾病自动化诊断解决方案主要集中在如抑郁等广泛疾病类别上的识别，尚不清楚患者自己可能都无法意识的动心理冲突能否从对话中自动识别。在本文中，我们提出AutoPsyC，这是第一个使用大规模语言模型（LLMs）从完整长度的操作化动心理诊断（OPD）访谈中识别动心理冲突存在及其意义的方法。我们的方法结合了参数高效微调和检索增强生成（RAG）的最新进展，并采用总结策略，有效处理整个90分钟长的对话。在包含141个诊断访谈的数据集上的评估表明，AutoPsyC在识别四项高度相关动心理冲突方面始终优于所有基线和消融条件。 

---
# JEEM: Vision-Language Understanding in Four Arabic Dialects 

**Title (ZH)**: JEEM: 四种阿拉伯方言的视觉-语言理解 

**Authors**: Karima Kadaoui, Hanin Atwany, Hamdan Al-Ali, Abdelrahman Mohamed, Ali Mekky, Sergei Tilga, Natalia Fedorova, Ekaterina Artemova, Hanan Aldarmaki, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2503.21910)  

**Abstract**: We introduce JEEM, a benchmark designed to evaluate Vision-Language Models (VLMs) on visual understanding across four Arabic-speaking countries: Jordan, The Emirates, Egypt, and Morocco. JEEM includes the tasks of image captioning and visual question answering, and features culturally rich and regionally diverse content. This dataset aims to assess the ability of VLMs to generalize across dialects and accurately interpret cultural elements in visual contexts. In an evaluation of five prominent open-source Arabic VLMs and GPT-4V, we find that the Arabic VLMs consistently underperform, struggling with both visual understanding and dialect-specific generation. While GPT-4V ranks best in this comparison, the model's linguistic competence varies across dialects, and its visual understanding capabilities lag behind. This underscores the need for more inclusive models and the value of culturally-diverse evaluation paradigms. 

**Abstract (ZH)**: JEEM：一种用于评估视觉-语言模型在阿拉伯语地区视觉理解能力的基准测试 

---
# Exponentially Weighted Instance-Aware Repeat Factor Sampling for Long-Tailed Object Detection Model Training in Unmanned Aerial Vehicles Surveillance Scenarios 

**Title (ZH)**: 基于无人机监控场景中长尾目标检测模型训练的指数加权实例感知重复因子采样方法 

**Authors**: Taufiq Ahmed, Abhishek Kumar, Constantino Álvarez Casado, Anlan Zhang, Tuomo Hänninen, Lauri Loven, Miguel Bordallo López, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2503.21893)  

**Abstract**: Object detection models often struggle with class imbalance, where rare categories appear significantly less frequently than common ones. Existing sampling-based rebalancing strategies, such as Repeat Factor Sampling (RFS) and Instance-Aware Repeat Factor Sampling (IRFS), mitigate this issue by adjusting sample frequencies based on image and instance counts. However, these methods are based on linear adjustments, which limit their effectiveness in long-tailed distributions. This work introduces Exponentially Weighted Instance-Aware Repeat Factor Sampling (E-IRFS), an extension of IRFS that applies exponential scaling to better differentiate between rare and frequent classes. E-IRFS adjusts sampling probabilities using an exponential function applied to the geometric mean of image and instance frequencies, ensuring a more adaptive rebalancing strategy. We evaluate E-IRFS on a dataset derived from the Fireman-UAV-RGBT Dataset and four additional public datasets, using YOLOv11 object detection models to identify fire, smoke, people and lakes in emergency scenarios. The results show that E-IRFS improves detection performance by 22\% over the baseline and outperforms RFS and IRFS, particularly for rare categories. The analysis also highlights that E-IRFS has a stronger effect on lightweight models with limited capacity, as these models rely more on data sampling strategies to address class imbalance. The findings demonstrate that E-IRFS improves rare object detection in resource-constrained environments, making it a suitable solution for real-time applications such as UAV-based emergency monitoring. 

**Abstract (ZH)**: Exponentially Weighted Instance-Aware Repeat Factor Sampling for Improving Rare Class Detection in Object Detection Models 

---
# StarFlow: Generating Structured Workflow Outputs From Sketch Images 

**Title (ZH)**: 星流：从素描图像生成结构化工作流输出 

**Authors**: Patrice Bechard, Chao Wang, Amirhossein Abaskohi, Juan Rodriguez, Christopher Pal, David Vazquez, Spandana Gella, Sai Rajeswar, Perouz Taslakian  

**Link**: [PDF](https://arxiv.org/pdf/2503.21889)  

**Abstract**: Workflows are a fundamental component of automation in enterprise platforms, enabling the orchestration of tasks, data processing, and system integrations. Despite being widely used, building workflows can be complex, often requiring manual configuration through low-code platforms or visual programming tools. To simplify this process, we explore the use of generative foundation models, particularly vision-language models (VLMs), to automatically generate structured workflows from visual inputs. Translating hand-drawn sketches or computer-generated diagrams into executable workflows is challenging due to the ambiguity of free-form drawings, variations in diagram styles, and the difficulty of inferring execution logic from visual elements. To address this, we introduce StarFlow, a framework for generating structured workflow outputs from sketches using vision-language models. We curate a diverse dataset of workflow diagrams -- including synthetic, manually annotated, and real-world samples -- to enable robust training and evaluation. We finetune and benchmark multiple vision-language models, conducting a series of ablation studies to analyze the strengths and limitations of our approach. Our results show that finetuning significantly enhances structured workflow generation, outperforming large vision-language models on this task. 

**Abstract (ZH)**: 基于视觉语言模型的草图到结构化工作流生成：StarFlow框架 

---
# RedditESS: A Mental Health Social Support Interaction Dataset -- Understanding Effective Social Support to Refine AI-Driven Support Tools 

**Title (ZH)**: RedditESS：一个心理健康社交支持互动数据集——理解有效的社交支持以优化AI驱动的支持工具 

**Authors**: Zeyad Alghamdi, Tharindu Kumarage, Garima Agrawal, Mansooreh Karami, Ibrahim Almuteb, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21888)  

**Abstract**: Effective mental health support is crucial for alleviating psychological distress. While large language model (LLM)-based assistants have shown promise in mental health interventions, existing research often defines "effective" support primarily in terms of empathetic acknowledgments, overlooking other essential dimensions such as informational guidance, community validation, and tangible coping strategies. To address this limitation and better understand what constitutes effective support, we introduce RedditESS, a novel real-world dataset derived from Reddit posts, including supportive comments and original posters' follow-up responses. Grounded in established social science theories, we develop an ensemble labeling mechanism to annotate supportive comments as effective or not and perform qualitative assessments to ensure the reliability of the annotations. Additionally, we demonstrate the practical utility of RedditESS by using it to guide LLM alignment toward generating more context-sensitive and genuinely helpful supportive responses. By broadening the understanding of effective support, our study paves the way for advanced AI-driven mental health interventions. 

**Abstract (ZH)**: 有效的心理健康支持对于缓解心理 distress 至关重要。尽管基于大规模语言模型 (LLM) 的助手在心理健康干预方面显示出前景，但现有研究往往主要从同理心认可的角度定义“有效”支持，忽视了其他重要的维度，如信息指导、社区验证和实际应对策略。为了解决这一局限性并更好地理解有效支持的含义，我们引入了 RedditESS，这是一个新颖的实际数据集，来源于 Reddit 发帖，包括支持性评论和原始发帖人的后续回复。基于现成的社会科学理论，我们开发了一种集成标注机制来标注有效和支持性或不支持性的评论，并进行定性评估以确保标注的可靠性。此外，我们通过利用 RedditESS 指引 LLM 向生成更具情境相关性和真正有用的支持性回应的方向发展。通过扩展对有效支持的理解，本研究为先进的 AI 驱动心理健康干预铺平了道路。 

---
# Foveated Instance Segmentation 

**Title (ZH)**: 注视点实例分割 

**Authors**: Hongyi Zeng, Wenxuan Liu, Tianhua Xia, Jinhui Chen, Ziyun Li, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21854)  

**Abstract**: Instance segmentation is essential for augmented reality and virtual reality (AR/VR) as it enables precise object recognition and interaction, enhancing the integration of virtual and real-world elements for an immersive experience. However, the high computational overhead of segmentation limits its application on resource-constrained AR/VR devices, causing large processing latency and degrading user experience. In contrast to conventional scenarios, AR/VR users typically focus on only a few regions within their field of view before shifting perspective, allowing segmentation to be concentrated on gaze-specific areas. This insight drives the need for efficient segmentation methods that prioritize processing instance of interest, reducing computational load and enhancing real-time performance. In this paper, we present a foveated instance segmentation (FovealSeg) framework that leverages real-time user gaze data to perform instance segmentation exclusively on instance of interest, resulting in substantial computational savings. Evaluation results show that FSNet achieves an IoU of 0.56 on ADE20K and 0.54 on LVIS, notably outperforming the baseline. The code is available at this https URL 

**Abstract (ZH)**: 注视点导向实例分割（FovealSeg）框架：基于实时用户注视数据的实例分割方法及其性能评估 

---
# Comparative Analysis of Image, Video, and Audio Classifiers for Automated News Video Segmentation 

**Title (ZH)**: 图像、视频和音频分类器的自动新闻视频分割比较分析 

**Authors**: Jonathan Attard, Dylan Seychell  

**Link**: [PDF](https://arxiv.org/pdf/2503.21848)  

**Abstract**: News videos require efficient content organisation and retrieval systems, but their unstructured nature poses significant challenges for automated processing. This paper presents a comprehensive comparative analysis of image, video, and audio classifiers for automated news video segmentation. This work presents the development and evaluation of multiple deep learning approaches, including ResNet, ViViT, AST, and multimodal architectures, to classify five distinct segment types: advertisements, stories, studio scenes, transitions, and visualisations. Using a custom-annotated dataset of 41 news videos comprising 1,832 scene clips, our experiments demonstrate that image-based classifiers achieve superior performance (84.34\% accuracy) compared to more complex temporal models. Notably, the ResNet architecture outperformed state-of-the-art video classifiers while requiring significantly fewer computational resources. Binary classification models achieved high accuracy for transitions (94.23\%) and advertisements (92.74\%). These findings advance the understanding of effective architectures for news video segmentation and provide practical insights for implementing automated content organisation systems in media applications. These include media archiving, personalised content delivery, and intelligent video search. 

**Abstract (ZH)**: 新闻视频需要高效的內容组织和检索系统，但由于其非结构化特性，为自动化处理带来了巨大挑战。本文对图像、视频和音频分类器在新闻视频自动分割中的应用进行了全面的比较分析。本文还阐述了多种深度学习方法的发展与评估，包括ResNet、ViViT、AST以及多模态架构，用于分类五大不同段落类型：广告、故事、演播室场景、过渡和可视化。使用包含41条新闻视频和1,832个场景片段的自标注数据集，我们的实验显示，基于图像的分类器在性能上优于更复杂的时间模型（准确率84.34%）。值得注意的是，ResNet架构在计算资源消耗显著减少的情况下，优于最先进的视频分类器。二元分类模型在过渡（94.23%）和广告（92.74%）分类上获得了高准确率。这些发现推进了对新闻视频分割有效架构的理解，并为媒体应用中的自动化内容组织系统实施提供了实用洞察，包括媒体归档、个性化内容交付和智能视频搜索。 

---
# ReCoM: Realistic Co-Speech Motion Generation with Recurrent Embedded Transformer 

**Title (ZH)**: ReCoM: 基于递归嵌入变压器的现实主义语音同步运动生成 

**Authors**: Yong Xie, Yunlian Sun, Hongwen Zhang, Yebin Liu, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21847)  

**Abstract**: We present ReCoM, an efficient framework for generating high-fidelity and generalizable human body motions synchronized with speech. The core innovation lies in the Recurrent Embedded Transformer (RET), which integrates Dynamic Embedding Regularization (DER) into a Vision Transformer (ViT) core architecture to explicitly model co-speech motion dynamics. This architecture enables joint spatial-temporal dependency modeling, thereby enhancing gesture naturalness and fidelity through coherent motion synthesis. To enhance model robustness, we incorporate the proposed DER strategy, which equips the model with dual capabilities of noise resistance and cross-domain generalization, thereby improving the naturalness and fluency of zero-shot motion generation for unseen speech inputs. To mitigate inherent limitations of autoregressive inference, including error accumulation and limited self-correction, we propose an iterative reconstruction inference (IRI) strategy. IRI refines motion sequences via cyclic pose reconstruction, driven by two key components: (1) classifier-free guidance improves distribution alignment between generated and real gestures without auxiliary supervision, and (2) a temporal smoothing process eliminates abrupt inter-frame transitions while ensuring kinematic continuity. Extensive experiments on benchmark datasets validate ReCoM's effectiveness, achieving state-of-the-art performance across metrics. Notably, it reduces the Fréchet Gesture Distance (FGD) from 18.70 to 2.48, demonstrating an 86.7% improvement in motion realism. Our project page is this https URL. 

**Abstract (ZH)**: 我们提出ReCoM，一种高效框架，用于生成与语音同步的高保真和可泛化的_human_body_动作。核心创新在于循环嵌入变换器（RET），它将动态嵌入正则化（DER）集成到Vision Transformer（ViT）核心架构中，以显式建模共言语动动态。该架构能够实现空间-时间依赖性建模，从而通过一致的动作合成增强手势的自然性和保真度。为了增强模型的鲁棒性，我们引入了提出的DER策略，使模型具备噪声抵抗和跨域泛化的双重能力，从而提高对未见过的语音输入的零样本动作生成的自然流畅度。为缓解自回归推理的固有限制，包括误差累积和自我纠正能力有限，我们提出了一种迭代重建推理（IRI）策略。IRI通过循环姿态重建细化动作序列，由两个关键组件驱动：（1）无分类引导提高生成手势和真实手势之间的分布对齐，无需辅助监督；（2）时间平滑过程消除帧间突变过渡，同时确保运动连贯性。基准数据集上的广泛实验证明了ReCoM的有效性，多项指标上取得了最佳性能。值得注意的是，它将Fréchet 动作距离（FGD）从18.70降低到2.48，显示出86.7%的动作真实感改进。我们的项目页面在此：https://xxxxxx。 

---
# LightSNN: Lightweight Architecture Search for Sparse and Accurate Spiking Neural Networks 

**Title (ZH)**: LightSNN：轻量级稀疏准确脉冲神经网络架构搜索 

**Authors**: Yesmine Abdennadher, Giovanni Perin, Riccardo Mazzieri, Jacopo Pegoraro, Michele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21846)  

**Abstract**: Spiking Neural Networks (SNNs) are highly regarded for their energy efficiency, inherent activation sparsity, and suitability for real-time processing in edge devices. However, most current SNN methods adopt architectures resembling traditional artificial neural networks (ANNs), leading to suboptimal performance when applied to SNNs. While SNNs excel in energy efficiency, they have been associated with lower accuracy levels than traditional ANNs when utilizing conventional architectures. In response, in this work we present LightSNN, a rapid and efficient Neural Network Architecture Search (NAS) technique specifically tailored for SNNs that autonomously leverages the most suitable architecture, striking a good balance between accuracy and efficiency by enforcing sparsity. Based on the spiking NAS network (SNASNet) framework, a cell-based search space including backward connections is utilized to build our training-free pruning-based NAS mechanism. Our technique assesses diverse spike activation patterns across different data samples using a sparsity-aware Hamming distance fitness evaluation. Thorough experiments are conducted on both static (CIFAR10 and CIFAR100) and neuromorphic datasets (DVS128-Gesture). Our LightSNN model achieves state-of-the-art results on CIFAR10 and CIFAR100, improves performance on DVS128Gesture by 4.49%, and significantly reduces search time, most notably offering a 98x speedup over SNASNet and running 30% faster than the best existing method on DVS128Gesture. 

**Abstract (ZH)**: 基于 stabbing 的轻量级神经网络架构搜索方法（LightSNN） 

---
# CMD-HAR: Cross-Modal Disentanglement for Wearable Human Activity Recognition 

**Title (ZH)**: CMD-HAR: 跨模态解耦的人体活动识别 

**Authors**: Hanyu Liu, Siyao Li, Ying Yu, Yixuan Jiang, Hang Xiao, Jingxi Long, Haotian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21843)  

**Abstract**: Human Activity Recognition (HAR) is a fundamental technology for numerous human - centered intelligent applications. Although deep learning methods have been utilized to accelerate feature extraction, issues such as multimodal data mixing, activity heterogeneity, and complex model deployment remain largely unresolved. The aim of this paper is to address issues such as multimodal data mixing, activity heterogeneity, and complex model deployment in sensor-based human activity recognition. We propose a spatiotemporal attention modal decomposition alignment fusion strategy to tackle the problem of the mixed distribution of sensor data. Key discriminative features of activities are captured through cross-modal spatio-temporal disentangled representation, and gradient modulation is combined to alleviate data heterogeneity. In addition, a wearable deployment simulation system is constructed. We conducted experiments on a large number of public datasets, demonstrating the effectiveness of the model. 

**Abstract (ZH)**: 基于传感器的人体活动识别中的多模态数据混叠、活动异质性和复杂模型部署问题研究 

---
# M-DocSum: Do LVLMs Genuinely Comprehend Interleaved Image-Text in Document Summarization? 

**Title (ZH)**: M-DocSum: LVLMs究竟在文档总结中真正理解了交错的图文信息吗？ 

**Authors**: Haolong Yan, Kaijun Tan, Yeqing Shen, Xin Huang, Zheng Ge, Xiangyu Zhang, Si Li, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21839)  

**Abstract**: We investigate a critical yet under-explored question in Large Vision-Language Models (LVLMs): Do LVLMs genuinely comprehend interleaved image-text in the document? Existing document understanding benchmarks often assess LVLMs using question-answer formats, which are information-sparse and difficult to guarantee the coverage of long-range dependencies. To address this issue, we introduce a novel and challenging Multimodal Document Summarization Benchmark (M-DocSum-Bench), which comprises 500 high-quality arXiv papers, along with interleaved multimodal summaries aligned with human preferences. M-DocSum-Bench is a reference-based generation task and necessitates the generation of interleaved image-text summaries using provided reference images, thereby simultaneously evaluating capabilities in understanding, reasoning, localization, and summarization within complex multimodal document scenarios. To facilitate this benchmark, we develop an automated framework to construct summaries and propose a fine-grained evaluation method called M-DocEval. Moreover, we further develop a robust summarization baseline, i.e., M-DocSum-7B, by progressive two-stage training with diverse instruction and preference data. The extensive results on our M-DocSum-Bench reveal that the leading LVLMs struggle to maintain coherence and accurately integrate information within long and interleaved contexts, often exhibiting confusion between similar images and a lack of robustness. Notably, M-DocSum-7B achieves state-of-the-art performance compared to larger and closed-source models (including GPT-4o, Gemini Pro, Claude-3.5-Sonnet and Qwen2.5-VL-72B, etc.), demonstrating the potential of LVLMs for improved interleaved image-text understanding. The code, data, and models are available at this https URL. 

**Abstract (ZH)**: 我们研究了一个在大规模视觉-语言模型（LVLMs）中关键但尚未充分探索的问题：LVLMs是否真正理解文档中的交错图像-文本？现有的文档理解基准通常使用问答格式评估LVLMs，这种格式信息稀疏且难以保证长距离依赖关系的覆盖。为解决这一问题，我们引入了一个新颖且具有挑战性的多模态文档总结基准（M-DocSum-Bench），该基准包含500份高质量的arXiv论文，并提供了与人类偏好对齐的交错多模态摘要。M-DocSum-Bench是一个基于参考的生成任务，要求使用提供的参考图像生成交错的图像-文本摘要，从而在复杂多模态文档场景中同时评估理解和推理、定位和摘要的能力。为了支持这一基准，我们开发了一种自动化框架来构建摘要，并提出了一种细粒度评估方法M-DocEval。此外，我们通过分阶段训练和多样化指令与偏好数据进一步开发了一个稳健的总结基线，即M-DocSum-7B。我们在M-DocSum-Bench上的广泛结果表明，领先的LVLMs在长且交错的上下文中难以保持连贯并准确整合信息，往往混淆相似的图像并缺乏鲁棒性。值得注意的是，M-DocSum-7B在与更大且封闭源模型（包括GPT-4o、Gemini Pro、Claude-3.5-Sonnet和Qwen2.5-VL-72B等）比较时，取得了最先进的性能，展示了LVLMs在交错图像-文本理解方面的潜力。相关代码、数据和模型可在以下链接获取。 

---
# MSPLoRA: A Multi-Scale Pyramid Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Title (ZH)**: MSPLoRA：一种多尺度金字塔低秩适应方法以实现高效的模型微调 

**Authors**: Jiancheng Zhao, Xingda Yu, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21838)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become an essential approach for adapting large-scale pre-trained models while reducing computational costs. Among PEFT methods, LoRA significantly reduces trainable parameters by decomposing weight updates into low-rank matrices. However, traditional LoRA applies a fixed rank across all layers, failing to account for the varying complexity of hierarchical information, which leads to inefficient adaptation and redundancy. To address this, we propose MSPLoRA (Multi-Scale Pyramid LoRA), which introduces Global Shared LoRA, Mid-Level Shared LoRA, and Layer-Specific LoRA to capture global patterns, mid-level features, and fine-grained information, respectively. This hierarchical structure reduces inter-layer redundancy while maintaining strong adaptation capability. Experiments on various NLP tasks demonstrate that MSPLoRA achieves more efficient adaptation and better performance while significantly reducing the number of trainable parameters. Furthermore, additional analyses based on Singular Value Decomposition validate its information decoupling ability, highlighting MSPLoRA as a scalable and effective optimization strategy for parameter-efficient fine-tuning in large language models. Our code is available at this https URL. 

**Abstract (ZH)**: 多尺度金字塔LoRA：一种高效的参数适配优化策略 

---
# A Multi-Modal Knowledge-Enhanced Framework for Vessel Trajectory Prediction 

**Title (ZH)**: 多模态知识增强框架船舶轨迹预测 

**Authors**: Haomin Yu, Tianyi Li, Kristian Torp, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21834)  

**Abstract**: Accurate vessel trajectory prediction facilitates improved navigational safety, routing, and environmental protection. However, existing prediction methods are challenged by the irregular sampling time intervals of the vessel tracking data from the global AIS system and the complexity of vessel movement. These aspects render model learning and generalization difficult. To address these challenges and improve vessel trajectory prediction, we propose the multi-modal knowledge-enhanced framework (MAKER) for vessel trajectory prediction. To contend better with the irregular sampling time intervals, MAKER features a Large language model-guided Knowledge Transfer (LKT) module that leverages pre-trained language models to transfer trajectory-specific contextual knowledge effectively. To enhance the ability to learn complex trajectory patterns, MAKER incorporates a Knowledge-based Self-paced Learning (KSL) module. This module employs kinematic knowledge to progressively integrate complex patterns during training, allowing for adaptive learning and enhanced generalization. Experimental results on two vessel trajectory datasets show that MAKER can improve the prediction accuracy of state-of-the-art methods by 12.08%-17.86%. 

**Abstract (ZH)**: 准确的船舶轨迹预测有助于提高航行安全、航线规划和环境保护。然而，现有的预测方法受到全球AIS系统中船舶跟踪数据不规则采样时间间隔以及船舶运动复杂性的挑战，这使得模型学习和泛化变得困难。为了应对这些挑战并改进船舶轨迹预测，我们提出了一个多模态知识增强框架（MAKER）用于船舶轨迹预测。MAKER通过一个大型语言模型指导的知识转移（LKT）模块，利用预训练语言模型有效转移轨迹特定上下文知识，以更好地应对不规则采样时间间隔。为了增强学习复杂轨迹模式的能力，MAKER还引入了基于知识的自适应学习（KSL）模块。该模块利用运动学知识在训练过程中逐步整合复杂模式，实现适应性学习和增强泛化。在两个船舶轨迹数据集上的实验结果表明，MAKER可以将最先进的方法的预测准确性提高12.08%-17.86%。 

---
# ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks 

**Title (ZH)**: ATP：适配阈值剪枝在量子神经网络高效数据编码中的应用 

**Authors**: Mohamed Afane, Gabrielle Ebbrecht, Ying Wang, Juntao Chen, Junaid Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2503.21815)  

**Abstract**: Quantum Neural Networks (QNNs) offer promising capabilities for complex data tasks, but are often constrained by limited qubit resources and high entanglement, which can hinder scalability and efficiency. In this paper, we introduce Adaptive Threshold Pruning (ATP), an encoding method that reduces entanglement and optimizes data complexity for efficient computations in QNNs. ATP dynamically prunes non-essential features in the data based on adaptive thresholds, effectively reducing quantum circuit requirements while preserving high performance. Extensive experiments across multiple datasets demonstrate that ATP reduces entanglement entropy and improves adversarial robustness when combined with adversarial training methods like FGSM. Our results highlight ATPs ability to balance computational efficiency and model resilience, achieving significant performance improvements with fewer resources, which will help make QNNs more feasible in practical, resource-constrained settings. 

**Abstract (ZH)**: 量子神经网络中的自适应阈值剪枝（Adaptive Threshold Pruning for Quantum Neural Networks） 

---
# Taxonomy Inference for Tabular Data Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行表格数据的分类学推断 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2503.21810)  

**Abstract**: Taxonomy inference for tabular data is a critical task of schema inference, aiming at discovering entity types (i.e., concepts) of the tables and building their hierarchy. It can play an important role in data management, data exploration, ontology learning, and many data-centric applications. Existing schema inference systems focus more on XML, JSON or RDF data, and often rely on lexical formats and structures of the data for calculating similarities, with limited exploitation of the semantics of the text across a table. Motivated by recent works on taxonomy completion and construction using Large Language Models (LLMs), this paper presents two LLM-based methods for taxonomy inference for tables: (i) EmTT which embeds columns by fine-tuning with contrastive learning encoder-alone LLMs like BERT and utilises clustering for hierarchy construction, and (ii) GeTT which generates table entity types and their hierarchy by iterative prompting using a decoder-alone LLM like GPT-4. Extensive evaluation on three real-world datasets with six metrics covering different aspects of the output taxonomies has demonstrated that EmTT and GeTT can both produce taxonomies with strong consistency relative to the Ground Truth. 

**Abstract (ZH)**: 基于大型语言模型的表格分类学推断方法 

---
# LERO: LLM-driven Evolutionary framework with Hybrid Rewards and Enhanced Observation for Multi-Agent Reinforcement Learning 

**Title (ZH)**: LEREO: 由大规模语言模型驱动的混合奖励及增强观测的多智能体强化学习演化框架 

**Authors**: Yuan Wei, Xiaohan Shan, Jianmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21807)  

**Abstract**: Multi-agent reinforcement learning (MARL) faces two critical bottlenecks distinct from single-agent RL: credit assignment in cooperative tasks and partial observability of environmental states. We propose LERO, a framework integrating Large language models (LLMs) with evolutionary optimization to address these MARL-specific challenges. The solution centers on two LLM-generated components: a hybrid reward function that dynamically allocates individual credit through reward decomposition, and an observation enhancement function that augments partial observations with inferred environmental context. An evolutionary algorithm optimizes these components through iterative MARL training cycles, where top-performing candidates guide subsequent LLM generations. Evaluations in Multi-Agent Particle Environments (MPE) demonstrate LERO's superiority over baseline methods, with improved task performance and training efficiency. 

**Abstract (ZH)**: 多智能体强化学习（MARL）面临两个关键瓶颈，不同于单智能体RL：协同任务中的信用分配和环境状态的部分可观测性。我们提出LERO框架，该框架结合大型语言模型（LLMs）和进化优化，以解决这些MARL特定挑战。解决方案集中在两个LLM生成的组件上：一种动态分配个体信用的混合奖励函数，以及通过推断环境上下文增强部分观察的观察增强函数。进化算法通过迭代的MARL训练周期优化这些组件，其中表现最佳的候选者指导后续LLM代的生成。LERO在多智能体粒子环境（MPE）中的评估表明其优于基线方法，在任务性能和训练效率方面均有提升。 

---
# Large Language Models Meet Contrastive Learning: Zero-Shot Emotion Recognition Across Languages 

**Title (ZH)**: 大型语言模型结合对比学习：跨语言零样本情绪识别 

**Authors**: Heqing Zou, Fengmao Lv, Desheng Zheng, Eng Siong Chng, Deepu Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21806)  

**Abstract**: Multilingual speech emotion recognition aims to estimate a speaker's emotional state using a contactless method across different languages. However, variability in voice characteristics and linguistic diversity poses significant challenges for zero-shot speech emotion recognition, especially with multilingual datasets. In this paper, we propose leveraging contrastive learning to refine multilingual speech features and extend large language models for zero-shot multilingual speech emotion estimation. Specifically, we employ a novel two-stage training framework to align speech signals with linguistic features in the emotional space, capturing both emotion-aware and language-agnostic speech representations. To advance research in this field, we introduce a large-scale synthetic multilingual speech emotion dataset, M5SER. Our experiments demonstrate the effectiveness of the proposed method in both speech emotion recognition and zero-shot multilingual speech emotion recognition, including previously unseen datasets and languages. 

**Abstract (ZH)**: 跨语言语音情感识别旨在通过无接触的方法，利用不同语言的语音特征来估计演讲者的情感状态。然而，语音特征的变异性与语言多样性对零样本多语言语音情感识别构成了重大挑战，尤其是在使用多语言数据集时。本文提出利用对比学习改进多语言语音特征并扩展大型语言模型，以实现零样本多语言语音情感估计。具体而言，我们采用一种新颖的两阶段训练框架来在情感空间中对齐语音信号与语言特征，捕捉情感感知和语言无关的语音表示。为了促进该领域的研究，我们引入了一个大规模合成多语言语音情感数据集M5SER。我们的实验表明，在语音情感识别和零样本多语言语音情感识别中，包括之前未见过的数据集和语言，所提出的方法均具有有效性。 

---
# ImF: Implicit Fingerprint for Large Language Models 

**Title (ZH)**: 隐式指纹：大型语言模型的隐式指纹 

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21805)  

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership. 

**Abstract (ZH)**: 大型语言模型（LLM）的训练耗资巨大且成本高，知识产权（IP）保护尤为重要。现有的大多数模型指纹方法通过向LLM中注入指纹来保护模型的所有权。这些方法生成的指纹对语义相关性弱，缺乏正常问答（QA）对在LLM中所具有的上下文连贯性和语义相关性。在本文中，我们提出了一种生成修订干预（GRI）攻击，能有效利用这一缺陷来删除指纹，从而突显了更安全的模型指纹方法的需求。因此，我们提出了一种新的注入指纹范式，称为隐式指纹（ImF）。ImF 构建具有强语义相关性的指纹对，并将其伪装成LLM中的自然QA对，确保指纹与正常模型行为一致，使其难以区分并对抗检测和删除。我们在多个LLM上的实验表明，在对抗条件下，ImF 保持了高验证成功率，提供了一种可靠的保护LLM所有权的解决方案。 

---
# Comparison of Metadata Representation Models for Knowledge Graph Embeddings 

**Title (ZH)**: 知识图嵌入中元数据表示模型的比较 

**Authors**: Shusaku Egami, Kyoumoto Matsushita, Takanori Ugai, Ken Fukuda  

**Link**: [PDF](https://arxiv.org/pdf/2503.21804)  

**Abstract**: Hyper-relational Knowledge Graphs (HRKGs) extend traditional KGs beyond binary relations, enabling the representation of contextual, provenance, and temporal information in domains, such as historical events, sensor data, video content, and narratives. HRKGs can be structured using several Metadata Representation Models (MRMs), including Reification (REF), Singleton Property (SGP), and RDF-star (RDR). However, the effects of different MRMs on KG Embedding (KGE) and Link Prediction (LP) models remain unclear. This study evaluates MRMs in the context of LP tasks, identifies the limitations of existing evaluation frameworks, and introduces a new task that ensures fair comparisons across MRMs. Furthermore, we propose a framework that effectively reflects the knowledge representations of the three MRMs in latent space. Experiments on two types of datasets reveal that REF performs well in simple HRKGs, whereas SGP is less effective. However, in complex HRKGs, the differences among MRMs in the LP tasks are minimal. Our findings contribute to an optimal knowledge representation strategy for HRKGs in LP tasks. 

**Abstract (ZH)**: 基于多个元数据表示模型的超关系知识图谱在链接预测任务中的评估与分析 

---
# Forecasting Volcanic Radiative Power (VPR) at Fuego Volcano Using Bayesian Regularized Neural Network 

**Title (ZH)**: 使用贝叶斯正则化神经网络预测危地马拉伊瓜托火山的火山辐射功率（VPR） 

**Authors**: Snehamoy Chatterjee, Greg Waite, Sidike Paheding, Luke Bowman  

**Link**: [PDF](https://arxiv.org/pdf/2503.21803)  

**Abstract**: Forecasting volcanic activity is critical for hazard assessment and risk mitigation. Volcanic Radiative Power (VPR), derived from thermal remote sensing data, serves as an essential indicator of volcanic activity. In this study, we employ Bayesian Regularized Neural Networks (BRNN) to predict future VPR values based on historical data from Fuego Volcano, comparing its performance against Scaled Conjugate Gradient (SCG) and Levenberg-Marquardt (LM) models. The results indicate that BRNN outperforms SCG and LM, achieving the lowest mean squared error (1.77E+16) and the highest R-squared value (0.50), demonstrating its superior ability to capture VPR variability while minimizing overfitting. Despite these promising results, challenges remain in improving the model's predictive accuracy. Future research should focus on integrating additional geophysical parameters, such as seismic and gas emission data, to enhance forecasting precision. The findings highlight the potential of machine learning models, particularly BRNN, in advancing volcanic activity forecasting, contributing to more effective early warning systems for volcanic hazards. 

**Abstract (ZH)**: 基于贝叶斯正则化神经网络的火山辐射功率预测及其应用： FUOGO 火山的案例研究 

---
# Efficient Joint Prediction of Multiple Future Tokens 

**Title (ZH)**: 高效联合预测多个未来词/token 

**Authors**: Kwangjun Ahn, Alex Lamb, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2503.21801)  

**Abstract**: In this short report, we introduce joint multi-token prediction (JTP), a lightweight modification of standard next-token prediction designed to enrich hidden state representations by jointly predicting multiple future tokens. Unlike previous multi-token prediction approaches, JTP strategically employs teacher forcing of future-tokens through a carefully designed representation bottleneck, allowing the model to encode rich predictive information with minimal computational overhead during training. We show that the JTP approach achieves a short-horizon belief state representation, while popular alternatives for multi-token prediction fail to do so. We demonstrate the effectiveness of our method on the synthetic star graph navigation task from from Bachmann and Nagarajan [2024], highlighting a significant performance improvement over existing methods. This manuscript presents promising preliminary results intended to stimulate further research. 

**Abstract (ZH)**: 一种轻量级的联合多令牌预测方法：实现短期信念状态表示的研究 

---
# ELM: Ensemble of Language Models for Predicting Tumor Group from Pathology Reports 

**Title (ZH)**: ELM：预测病理报告中肿瘤类型的语言模型集成 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21800)  

**Abstract**: Population-based cancer registries (PBCRs) face a significant bottleneck in manually extracting data from unstructured pathology reports, a process crucial for tasks like tumor group assignment, which can consume 900 person-hours for approximately 100,000 reports. To address this, we introduce ELM (Ensemble of Language Models), a novel ensemble-based approach leveraging both small language models (SLMs) and large language models (LLMs). ELM utilizes six fine-tuned SLMs, where three SLMs use the top part of the pathology report and three SLMs use the bottom part. This is done to maximize report coverage. ELM requires five-out-of-six agreement for a tumor group classification. Disagreements are arbitrated by an LLM with a carefully curated prompt. Our evaluation across nineteen tumor groups demonstrates ELM achieves an average precision and recall of 0.94, outperforming single-model and ensemble-without-LLM approaches. Deployed at the British Columbia Cancer Registry, ELM demonstrates how LLMs can be successfully applied in a PBCR setting to achieve state-of-the-art results and significantly enhance operational efficiencies, saving hundreds of person-hours annually. 

**Abstract (ZH)**: 基于人群的癌症注册库中语言模型集成方法（ELM）在病理报告解析中的应用：一种处理肿瘤分组的关键任务的高效解决方案 

---
# A Novel Two-Phase Cooperative Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping 

**Title (ZH)**: 一种新型两阶段协同演化框架，用于具有复杂重叠的大规模全局优化 

**Authors**: Wenjie Qiu, Hongshu Guo, Zeyuan Ma, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.21797)  

**Abstract**: Cooperative Co-evolution, through the decomposition of the problem space, is a primary approach for solving large-scale global optimization problems. Typically, when the subspaces are disjoint, the algorithms demonstrate significantly both effectiveness and efficiency compared to non-decomposition algorithms. However, the presence of overlapping variables complicates the decomposition process and adversely affects the performance of cooperative co-evolution. In this study, we propose a novel two-phase cooperative co-evolution framework to address large-scale global optimization problems with complex overlapping. An effective method for decomposing overlapping problems, grounded in their mathematical properties, is embedded within the framework. Additionally, a customizable benchmark for overlapping problems is introduced to extend existing benchmarks and facilitate experimentation. Extensive experiments demonstrate that the algorithm instantiated within our framework significantly outperforms existing algorithms. The results reveal the characteristics of overlapping problems and highlight the differing strengths of cooperative co-evolution and non-decomposition algorithms. Our work is open-source and accessible at: this https URL. 

**Abstract (ZH)**: 通过问题空间分解实现的合作共进化是解决大规模全局优化问题的主要方法。当子空间不相交时，该方法相较于非分解算法在有效性和效率上表现出显著优势。然而，重叠变量的存在使分解过程复杂化，并影响合作共进化的性能。在本研究中，我们提出了一种针对具有复杂重叠问题的新两阶段合作共进化框架，并基于其数学特性提出了一个有效的分解方法。此外，我们引入了一个可定制的重叠问题基准，以扩展现有基准并促进实验。广泛的实验表明，嵌入在该框架中的算法显著优于现有算法。结果揭示了重叠问题的特性，并突显了合作共进化和非分解算法的不同优势。我们的工作是开源的，可通过以下链接访问：this https URL。 

---
# Threshold Adaptation in Spiking Networks Enables Shortest Path Finding and Place Disambiguation 

**Title (ZH)**: 阈值自适应在脉冲神经网络中的实现 enables 最短路径寻找和位置歧义消解 

**Authors**: Robin Dietrich, Tobias Fischer, Nicolai Waniek, Nico Reeb, Michael Milford, Alois Knoll, Adam D. Hines  

**Link**: [PDF](https://arxiv.org/pdf/2503.21795)  

**Abstract**: Efficient spatial navigation is a hallmark of the mammalian brain, inspiring the development of neuromorphic systems that mimic biological principles. Despite progress, implementing key operations like back-tracing and handling ambiguity in bio-inspired spiking neural networks remains an open challenge. This work proposes a mechanism for activity back-tracing in arbitrary, uni-directional spiking neuron graphs. We extend the existing replay mechanism of the spiking hierarchical temporal memory (S-HTM) by our spike timing-dependent threshold adaptation (STDTA), which enables us to perform path planning in networks of spiking neurons. We further present an ambiguity dependent threshold adaptation (ADTA) for identifying places in an environment with less ambiguity, enhancing the localization estimate of an agent. Combined, these methods enable efficient identification of the shortest path to an unambiguous target. Our experiments show that a network trained on sequences reliably computes shortest paths with fewer replays than the steps required to reach the target. We further show that we can identify places with reduced ambiguity in multiple, similar environments. These contributions advance the practical application of biologically inspired sequential learning algorithms like the S-HTM towards neuromorphic localization and navigation. 

**Abstract (ZH)**: 高效的空间导航是哺乳动物大脑的一个典型特征，激发了模仿生物原理的神经形态系统的开发。尽管取得了进展，但在生物启发的脉冲神经网络中实现关键操作如回溯追踪和处理歧义仍然是一个开放的挑战。本工作提出了一种机制，用于在任意单向脉冲神经元图中执行活动回溯追踪。我们扩展了现有的脉冲层次时间记忆(S-HTM)的回放机制，引入了基于 spike 时间相关的阈值适应(STDTA)，这使我们能够在脉冲神经元网络中执行路径规划。我们进一步提出了基于歧义的阈值适应(ADTA)，用于识别环境中的歧义较少的地方，从而增强代理的定位估计。结合这两种方法，可以高效地识别到一个明确目标的最短路径。实验结果显示，一个在序列上训练的网络能够通过较少的回放可靠地计算出最短路径，直到目标。我们还展示了在多个相似环境中识别歧义较少的地方的能力。这些贡献推动了基于S-HTM等生物启发的序列学习算法的实际应用，向着神经形态定位和导航方向发展。 

---
# Architecture of Information 

**Title (ZH)**: 信息架构 

**Authors**: Yurii Parzhyn  

**Link**: [PDF](https://arxiv.org/pdf/2503.21794)  

**Abstract**: The paper explores an approach to constructing energy landscapes of a formal neuron and multilayer artificial neural networks (ANNs). Their analysis makes it possible to determine the conceptual limitations of both classification ANNs (e.g., MLP or CNN) and generative ANN models. The study of informational and thermodynamic entropy in formal neuron and ANN models leads to the conclusion about the energetic nature of informational entropy. The application of the Gibbs free energy concept allows representing the output information of ANNs as the structured part of enthalpy. Modeling ANNs as energy systems makes it possible to interpret the structure of their internal energy as an internal model of the external world, which self-organizes based on the interaction of the system's internal energy components. The control of the self-organization and evolution process of this model is carried out through an energy function (analogous to the Lyapunov function) based on reduction operators. This makes it possible to introduce a new approach to constructing self-organizing and evolutionary ANNs with direct learning, which does not require additional external algorithms. The presented research makes it possible to formulate a formal definition of information in terms of the interaction processes between the internal and external energy of the system. 

**Abstract (ZH)**: 论文探讨了构建形式神经元和多层人工神经网络（ANNs）能量景观的方法。对该模型的分析有助于确定分类ANN（如MLP或CNN）和生成性ANN模型的概念限制。通过研究形式神经元和ANN模型中的信息熵和热力学熵，得出信息熵的能量性质结论。利用吉布斯自由能的概念可以将ANN的输出信息表示为焓的有序部分。将ANN建模为能量系统，使其内部能量结构能够解释为对外部世界的内部模型，并通过系统内部能量组件的相互作用自我组织。通过对能量函数（类似于李雅普un夫函数）的控制操作来调控此模型的自我组织和进化过程。这使得可以直接学习构建自我组织和进化的ANN的新方法得以引入，无需额外的外部算法。本研究使我们能够用系统内外能之间的相互作用过程来给出信息的形式定义。 

---
# Input-Triggered Hardware Trojan Attack on Spiking Neural Networks 

**Title (ZH)**: 基于输入触发的硬件木马攻击在神经脉冲网络中 

**Authors**: Spyridon Raptis, Paul Kling, Ioannis Kaskampas, Ihsen Alouani, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.21793)  

**Abstract**: Neuromorphic computing based on spiking neural networks (SNNs) is emerging as a promising alternative to traditional artificial neural networks (ANNs), offering unique advantages in terms of low power consumption. However, the security aspect of SNNs is under-explored compared to their ANN counterparts. As the increasing reliance on AI systems comes with unique security risks and challenges, understanding the vulnerabilities and threat landscape is essential as neuromorphic computing matures. In this effort, we propose a novel input-triggered Hardware Trojan (HT) attack for SNNs. The HT mechanism is condensed in the area of one neuron. The trigger mechanism is an input message crafted in the spiking domain such that a selected neuron produces a malicious spike train that is not met in normal settings. This spike train triggers a malicious modification in the neuron that forces it to saturate, firing permanently and failing to recover to its resting state even when the input activity stops. The excessive spikes pollute the network and produce misleading decisions. We propose a methodology to select an appropriate neuron and to generate the input pattern that triggers the HT payload. The attack is illustrated by simulation on three popular benchmarks in the neuromorphic community. We also propose a hardware implementation for an analog spiking neuron and a digital SNN accelerator, demonstrating that the HT has a negligible area and power footprint and, thereby, can easily evade detection. 

**Abstract (ZH)**: 基于突触神经网络（SNNs）的神经形态计算正在成为传统人工神经网络（ANNs）的有前途的替代方案，提供了独特的低功耗优势。然而，与ANNs相比，SNNs的安全性方面尚未得到充分探索。随着对人工智能系统的依赖性增加，带来了独特的安全风险和挑战，因此，在神经形态计算成熟过程中理解其脆弱性和威胁 landscape 至关重要。为此，我们提出了一种新颖的输入触发硬件 Trojan（HT）攻击方法，该方法将 HT 机制集中在单个神经元的区域。触发机制是在突触域中精心制作的输入消息，使得选定的神经元产生一种在正常情况下不会出现的恶意尖峰序列。这种尖峰序列触发了神经元的恶意修改，使其饱和，永久放电，并且即使输入活动停止也无法恢复到静息状态。过多的尖峰污染了网络并产生了误导性的决策。我们提出了一种方法来选择合适的神经元并生成触发 HT 载荷的输入模式。通过模拟在神经形态社区中流行的三个基准，展示了该攻击方法。此外，我们提出了一种模拟突触神经元和数字 SNN 加速器的硬件实现，证明了 HT 几乎没有面积和功耗开销，从而可以轻松规避检测。 

---
# March Madness Tournament Predictions Model: A Mathematical Modeling Approach 

**Title (ZH)**: March Madnesstournament预测模型：一种数学建模方法 

**Authors**: Christian McIver, Karla Avalos, Nikhil Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2503.21790)  

**Abstract**: This paper proposes a model to predict the outcome of the March Madness tournament based on historical NCAA basketball data since 2013. The framework of this project is a simplification of the FiveThrityEight NCAA March Madness prediction model, where the only four predictors of interest are Adjusted Offensive Efficiency (ADJOE), Adjusted Defensive Efficiency (ADJDE), Power Rating, and Two-Point Shooting Percentage Allowed. A logistic regression was utilized with the aforementioned metrics to generate a probability of a particular team winning each game. Then, a tournament simulation is developed and compared to real-world March Madness brackets to determine the accuracy of the model. Accuracies of performance were calculated using a naive approach and a Spearman rank correlation coefficient. 

**Abstract (ZH)**: 基于2013年以来NCAA历史篮球数据的March Madnesstournament结果预测模型 

---
# From Deep Learning to LLMs: A survey of AI in Quantitative Investment 

**Title (ZH)**: 从深度学习到大语言模型：量化投资中人工智能的综述 

**Authors**: Bokai Cao, Saizhuo Wang, Xinyi Lin, Xiaojun Wu, Haohan Zhang, Lionel M. Ni, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.21422)  

**Abstract**: Quantitative investment (quant) is an emerging, technology-driven approach in asset management, increasingy shaped by advancements in artificial intelligence. Recent advances in deep learning and large language models (LLMs) for quant finance have improved predictive modeling and enabled agent-based automation, suggesting a potential paradigm shift in this field. In this survey, taking alpha strategy as a representative example, we explore how AI contributes to the quantitative investment pipeline. We first examine the early stage of quant research, centered on human-crafted features and traditional statistical models with an established alpha pipeline. We then discuss the rise of deep learning, which enabled scalable modeling across the entire pipeline from data processing to order execution. Building on this, we highlight the emerging role of LLMs in extending AI beyond prediction, empowering autonomous agents to process unstructured data, generate alphas, and support self-iterative workflows. 

**Abstract (ZH)**: 量化投资（Quant）是一种新兴的技术驱动的资产管理模式，日益受到人工智能进步的影响。量化金融中深度学习和大语言模型（LLMs）的最新进展提高了预测建模能力，并使基于代理的自动化成为可能，这可能在该领域引发范式转变。在本文综述中，以阿尔法策略为例，我们探讨人工智能如何贡献于量化投资流程。我们首先研究量化研究的早期阶段，集中在人工构建的特征和传统统计模型，并具有成熟的阿尔法流程。接着讨论深度学习的兴起，这一技术使从数据处理到下单执行的整个流程中可扩展的建模成为可能。在此基础上，我们强调大语言模型在扩展人工智能范围方面的作用，使自主代理能够处理非结构化数据、生成阿尔法并支持自迭代工作流程。 

---
# Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos 

**Title (ZH)**: Mamba-3D作为遮蔽自编码器用于医学超声视频的准确和数据高效分析 

**Authors**: Jiaheng Zhou, Yanfeng Zhou, Wei Fang, Yuxing Tang, Le Lu, Ge Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20258)  

**Abstract**: Ultrasound videos are an important form of clinical imaging data, and deep learning-based automated analysis can improve diagnostic accuracy and clinical efficiency. However, the scarcity of labeled data and the inherent challenges of video analysis have impeded the advancement of related methods. In this work, we introduce E-ViM$^3$, a data-efficient Vision Mamba network that preserves the 3D structure of video data, enhancing long-range dependencies and inductive biases to better model space-time correlations. With our design of Enclosure Global Tokens (EGT), the model captures and aggregates global features more effectively than competing methods. To further improve data efficiency, we employ masked video modeling for self-supervised pre-training, with the proposed Spatial-Temporal Chained (STC) masking strategy designed to adapt to various video scenarios. Experiments demonstrate that E-ViM$^3$ performs as the state-of-the-art in two high-level semantic analysis tasks across four datasets of varying sizes: EchoNet-Dynamic, CAMUS, MICCAI-BUV, and WHBUS. Furthermore, our model achieves competitive performance with limited labels, highlighting its potential impact on real-world clinical applications. 

**Abstract (ZH)**: 基于超声视频的E-ViM$^3$数据高效视网膜网络及其在空间-时间相关性建模中的应用 

---
