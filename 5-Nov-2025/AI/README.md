# Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything 

**Title (ZH)**: Agent-Omni: 测试时跨模态推理的理解通用模型协调方法 

**Authors**: Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.02834)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）展现出了强大的能力，但仍局限于固定的模态对，并且需要昂贵的调整与大规模对齐的数据集。构建能够整合文本、图像、音频和视频的全能模型仍然不切实际且缺乏 robust 的推理支持。本文提出了一种 Agent-Omni 框架，通过主代理系统协调现有的基础模型，使多模态推理能够在无需重新训练的情况下变得灵活。主代理解释用户意图，将子任务委托给特定于模态的代理，并将它们的输出整合为连贯的响应。在跨文本、图像、音频、视频和全能基准上的广泛实验表明，Agent-Omni 在需要复杂跨模态推理的任务中始终能够实现最先进的性能。其基于代理的设计使得能够无缝集成专门的基础模型，确保对不同输入的适应性同时保持透明性和可解释性。此外，该框架是模块化的且易于扩展，可以让未来的改进随着更强的模型变得可用而进行。我们开源了实现，以支持对可扩展和可靠的全能推理的持续研究。 

---
# Neurosymbolic Deep Learning Semantics 

**Title (ZH)**: 神经符号深度学习语义 

**Authors**: Artur d'Avila Garcez, Simon Odense  

**Link**: [PDF](https://arxiv.org/pdf/2511.02825)  

**Abstract**: Artificial Intelligence (AI) is a powerful new language of science as evidenced by recent Nobel Prizes in chemistry and physics that recognized contributions to AI applied to those areas. Yet, this new language lacks semantics, which makes AI's scientific discoveries unsatisfactory at best. With the purpose of uncovering new facts but also improving our understanding of the world, AI-based science requires formalization through a framework capable of translating insight into comprehensible scientific knowledge. In this paper, we argue that logic offers an adequate framework. In particular, we use logic in a neurosymbolic framework to offer a much needed semantics for deep learning, the neural network-based technology of current AI. Deep learning and neurosymbolic AI lack a general set of conditions to ensure that desirable properties are satisfied. Instead, there is a plethora of encoding and knowledge extraction approaches designed for particular cases. To rectify this, we introduced a framework for semantic encoding, making explicit the mapping between neural networks and logic, and characterizing the common ingredients of the various existing approaches. In this paper, we describe succinctly and exemplify how logical semantics and neural networks are linked through this framework, we review some of the most prominent approaches and techniques developed for neural encoding and knowledge extraction, provide a formal definition of our framework, and discuss some of the difficulties of identifying a semantic encoding in practice in light of analogous problems in the philosophy of mind. 

**Abstract (ZH)**: 人工智能：作为科学新语言的逻辑框架 

---
# Kosmos: An AI Scientist for Autonomous Discovery 

**Title (ZH)**: Kosmos: 人工科学家实现自主发现 

**Authors**: Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Daniel L. Barabasi, Siddharth Narayanan, Nicky Evans, Shriya Reddy, Martha Foiani, Aizad Kamal, Leah P. Shriver, Fang Cao, Asmamaw T. Wassie, Jon M. Laurent, Edwin Melville-Green, Mayk Caldas, Albert Bou, Kaleigh F. Roberts, Sladjana Zagorac, Timothy C. Orr, Miranda E. Orr, Kevin J. Zwezdaryk, Ali E. Ghareeb, Laurie McCoy, Bruna Gomes, Euan A. Ashley, Karen E. Duff, Tonio Buonassisi, Tom Rainforth, Randall J. Bateman, Michael Skarlinski, Samuel G. Rodriques, Michaela M. Hinks, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2511.02824)  

**Abstract**: Data-driven scientific discovery requires iterative cycles of literature search, hypothesis generation, and data analysis. Substantial progress has been made towards AI agents that can automate scientific research, but all such agents remain limited in the number of actions they can take before losing coherence, thus limiting the depth of their findings. Here we present Kosmos, an AI scientist that automates data-driven discovery. Given an open-ended objective and a dataset, Kosmos runs for up to 12 hours performing cycles of parallel data analysis, literature search, and hypothesis generation before synthesizing discoveries into scientific reports. Unlike prior systems, Kosmos uses a structured world model to share information between a data analysis agent and a literature search agent. The world model enables Kosmos to coherently pursue the specified objective over 200 agent rollouts, collectively executing an average of 42,000 lines of code and reading 1,500 papers per run. Kosmos cites all statements in its reports with code or primary literature, ensuring its reasoning is traceable. Independent scientists found 79.4% of statements in Kosmos reports to be accurate, and collaborators reported that a single 20-cycle Kosmos run performed the equivalent of 6 months of their own research time on average. Furthermore, collaborators reported that the number of valuable scientific findings generated scales linearly with Kosmos cycles (tested up to 20 cycles). We highlight seven discoveries made by Kosmos that span metabolomics, materials science, neuroscience, and statistical genetics. Three discoveries independently reproduce findings from preprinted or unpublished manuscripts that were not accessed by Kosmos at runtime, while four make novel contributions to the scientific literature. 

**Abstract (ZH)**: 数据驱动的科学发现需要迭代循环的文献搜索、假设生成和数据分析。尽管已经取得进展，能够自动进行科学研究的AI代理仍受限于在失去连贯性前能执行的动作数量，从而限制了其发现的深度。我们介绍了Kosmos，一种自动进行数据驱动发现的AI科学家。给定一个开放性的目标和一个数据集，Kosmos运行最多12小时，执行数据分析、文献搜索和假设生成的循环，然后将发现综合成科学报告。与先前的系统不同，Kosmos使用结构化的世界模型在数据处理代理和文献搜索代理之间共享信息。世界模型使Kosmos能够在200次代理演练中连贯地追求指定的目标，合计执行平均42,000行代码，并在每次运行中阅读1,500篇论文。Kosmos在报告中的所有声明均引用代码或原始文献，确保其推理可追踪。独立科学家发现Kosmos报告中的79.4%的陈述是准确的，合作者报告说每次20循环的Kosmos运行相当于他们的6个月研究时间。此外，合作者报告说，有价值的科学发现的数量与Kosmos的循环次数呈线性增长（测试到20次循环）。我们强调了Kosmos在代谢组学、材料科学、神经科学和统计遗传学等领域做出的七个发现。其中三个发现独立重现了预打印或未发表手稿中的发现，而另外四个为科学文献做出了新颖的贡献。 

---
# Optimizing AI Agent Attacks With Synthetic Data 

**Title (ZH)**: 使用合成数据优化AI代理攻击 

**Authors**: Chloe Loughridge, Paul Colognese, Avery Griffin, Tyler Tracy, Jon Kutasov, Joe Benton  

**Link**: [PDF](https://arxiv.org/pdf/2511.02823)  

**Abstract**: As AI deployments become more complex and high-stakes, it becomes increasingly important to be able to estimate their risk. AI control is one framework for doing so. However, good control evaluations require eliciting strong attack policies. This can be challenging in complex agentic environments where compute constraints leave us data-poor. In this work, we show how to optimize attack policies in SHADE-Arena, a dataset of diverse realistic control environments. We do this by decomposing attack capability into five constituent skills -- suspicion modeling, attack selection, plan synthesis, execution and subtlety -- and optimizing each component individually. To get around the constraint of limited data, we develop a probabilistic model of attack dynamics, optimize our attack hyperparameters using this simulation, and then show that the results transfer to SHADE-Arena. This results in a substantial improvement in attack strength, reducing safety score from a baseline of 0.87 to 0.41 using our scaffold. 

**Abstract (ZH)**: 随着AI部署变得越来越复杂且高风险，估算其风险的重要性日益增加。AI控制是一种为此目的的框架。然而，良好的控制评估需要引出有效的攻击策略。在计算资源有限而数据匮乏的复杂代理环境中，这颇具挑战性。在本研究中，我们展示了如何在SHADE-Arena数据集中优化攻击策略，SHADE-Arena包含多样化的现实控制环境。我们通过将攻击能力分解为五大组成部分——疑点建模、攻击选择、计划合成、执行和细腻度——并对每个部分分别进行优化来实现这一点。为了应对数据有限的约束，我们开发了一个攻击动态的概率模型，并使用该模型优化攻击超参数，然后证明这种方法可以应用于SHADE-Arena，结果显著提高了攻击强度，使用我们的构造方法将安全评分从基线的0.87降低到0.41。 

---
# Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning 

**Title (ZH)**: Orion-MSP: 多尺度稀疏注意力机制在表格形式上下文学习中的应用 

**Authors**: Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02818)  

**Abstract**: Tabular data remain the predominant format for real-world applications. Yet, developing effective neural models for tabular data remains challenging due to heterogeneous feature types and complex interactions occurring at multiple scales. Recent advances in tabular in-context learning (ICL), such as TabPFN and TabICL, have achieved state-of-the-art performance comparable to gradient-boosted trees (GBTs) without task-specific fine-tuning. However, current architectures exhibit key limitations: (1) single-scale feature processing that overlooks hierarchical dependencies, (2) dense attention with quadratic scaling in table width, and (3) strictly sequential component processing that prevents iterative representation refinement and cross-component communication. To address these challenges, we introduce Orion-MSP, a tabular ICL architecture featuring three key innovations: (1) multi-scale processing to capture hierarchical feature interactions; (2) block-sparse attention combining windowed, global, and random patterns for scalable efficiency and long-range connectivity; and (3) a Perceiver-style memory enabling safe bidirectional information flow across components. Across diverse benchmarks, Orion-MSP matches or surpasses state-of-the-art performance while scaling effectively to high-dimensional tables, establishing a new standard for efficient tabular in-context learning. The model is publicly available at this https URL . 

**Abstract (ZH)**: 表格数据仍然是现实世界应用中的主要格式。然而，由于特征类型异构和多尺度复杂相互作用的存在，开发有效的神经模型来处理表格数据仍然具有挑战性。最近在表格上下文学习（ICL）方面的进展，如TabPFN和TabICL，已经实现了与梯度提升树（GBTs）相媲美的性能，且无需针对特定任务进行微调。然而，当前的架构存在几个关键限制：（1）单一尺度的特征处理，忽视了层次依赖性；（2）密集注意力机制，时间复杂度随表格宽度平方增长；（3）严格顺序的组件处理，不利于迭代表示精炼和跨组件通信。为了解决这些挑战，我们引入了Orion-MSP，这是一种具有三个关键创新的表格ICL架构：（1）多尺度处理以捕获层次特征相互作用；（2）块稀疏注意力结合窗口局部、全局及随机模式，以实现高效性和长距离连接；（3）类似Perceiver的记忆机制，实现组件间安全的双向信息流。Orion-MSP在多种基准测试中实现了或超越了最先进性能，同时能有效扩展到高维表格，确立了高效表格ICL的新标准。模型已在以下链接公开：this https URL。 

---
# When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning 

**Title (ZH)**: 一种诊断视角下的多模态推理：当一种模态损害其他模态时 

**Authors**: Chenyu Zhang, Minsol Kim, Shohreh Ghorbani, Jingyao Wu, Rosalind Picard, Patricia Maes, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02794)  

**Abstract**: Despite rapid growth in multimodal large language models (MLLMs), their reasoning traces remain opaque: it is often unclear which modality drives a prediction, how conflicts are resolved, or when one stream dominates. In this paper, we introduce modality sabotage, a diagnostic failure mode in which a high-confidence unimodal error overrides other evidence and misleads the fused result. To analyze such dynamics, we propose a lightweight, model-agnostic evaluation layer that treats each modality as an agent, producing candidate labels and a brief self-assessment used for auditing. A simple fusion mechanism aggregates these outputs, exposing contributors (modalities supporting correct outcomes) and saboteurs (modalities that mislead). Applying our diagnostic layer in a case study on multimodal emotion recognition benchmarks with foundation models revealed systematic reliability profiles, providing insight into whether failures may arise from dataset artifacts or model limitations. More broadly, our framework offers a diagnostic scaffold for multimodal reasoning, supporting principled auditing of fusion dynamics and informing possible interventions. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）迅速增长，但其推理过程仍然不透明：往往是不清楚哪种模态驱动预测、冲突如何解决，或哪种模态占主导地位。本文引入了模态破坏这一诊断失效模式，其中高置信度的单模态错误覆盖其他证据，误导融合结果。为分析这种动态，我们提出了一个轻量级、模型无关的评估层，将每个模态视为代理，生成候选标签和简短的自我评估，用于审计。简单的融合机制汇总这些输出，揭示贡献者（支持正确结果的模态）和破坏者（误导模态）。在基础模型上对多模态情绪识别基准进行案例研究，揭示了系统性的可靠性概况，为是否源于数据集缺陷或模型限制提供了见解。更广泛而言，我们的框架为多模态推理提供了一种诊断支撑结构，支持融合动态的原理性审计，并指导可能的干预措施。 

---
# LLM-Supported Formal Knowledge Representation for Enhancing Control Engineering Content with an Interactive Semantic Layer 

**Title (ZH)**: 基于LLM的支持形式化知识表示以增强交互语义层的控制工程内容 

**Authors**: Julius Fiedler, Carsten Knoll, Klaus Röbenack  

**Link**: [PDF](https://arxiv.org/pdf/2511.02759)  

**Abstract**: The rapid growth of research output in control engineering calls for new approaches to structure and formalize domain knowledge. This paper briefly describes an LLM-supported method for semi-automated generation of formal knowledge representations that combine human readability with machine interpretability and increased expressiveness. Based on the Imperative Representation of Knowledge (PyIRK) framework, we demonstrate how language models can assist in transforming natural-language descriptions and mathematical definitions (available as LaTeX source code) into a formalized knowledge graph. As a first application we present the generation of an ``interactive semantic layer'' to enhance the source documents in order to facilitate knowledge transfer. From our perspective this contributes to the vision of easily accessible, collaborative, and verifiable knowledge bases for the control engineering domain. 

**Abstract (ZH)**: 控制工程领域研究成果的快速发展需要新的方法来结构化和形式化领域知识。本文简要描述了一种基于LLM的支持半自动化生成结合人类可读性和机器可解释性的形式知识表示的方法。基于 Imperative Representation of Knowledge (PyIRK) 框架，我们展示了语言模型如何辅助将自然语言描述和数学定义（以LaTeX源代码形式提供）转换为形式化的知识图谱。作为第一个应用，我们展示了生成一个“交互式语义层”的方法，以增强源文档，从而促进知识转移。从我们的角度看，这有助于实现控制工程领域易于访问、合作共享和可验证的知识库的愿景。 

---
# Using Span Queries to Optimize for Cache and Attention Locality 

**Title (ZH)**: 使用区间查询优化缓存和注意力局部性 

**Authors**: Paul Castro, Nick Mitchell, Nathan Ordonez, Thomas Parnell, Mudhakar Srivatsa, Antoni Viros i Martin  

**Link**: [PDF](https://arxiv.org/pdf/2511.02749)  

**Abstract**: Clients are evolving beyond chat completion, and now include a variety of innovative inference-time scaling and deep reasoning techniques. At the same time, inference servers remain heavily optimized for chat completion. Prior work has shown that large improvements to KV cache hit rate are possible if inference servers evolve towards these non-chat use cases. However, they offer solutions that are also optimized for a single use case, RAG. In this paper, we introduce the span query to generalize the interface to the inference server. We demonstrate that chat, RAG, inference-time scaling, and agentic workloads can all be expressed as span queries. We show how the critical distinction that had been assumed by prior work lies in whether the order of the inputs matter -- do they commute? In chat, they do not. In RAG, they often do. This paper introduces span queries, which are expression trees of inference calls, linked together with commutativity constraints. We describe span query syntax and semantics. We show how they can be automatically optimized to improve KV cache locality. We show how a small change to vLLM (affecting only 492 lines) can enable high-performance execution of span queries. Using this stack, we demonstrate that span queries can achieve 10-20x reductions in TTFT for two distinct non-chat use cases. Finally, we show that span queries can also be optimized to improve attention locality, so as to avoid the so-called lost-in-the-middle problem. We demonstrate that an attention-optimized span query on a 2b parameter model vastly outperforms the accuracy of a stock inference server using an 8b model. 

**Abstract (ZH)**: 客户端超越了单一聊天完成，现在包括各种创新的推理时扩展和深度推理技术。与此同时，推理服务器仍然 heavily 优化用于聊天完成。先前的工作表明，如果推理服务器向这些非聊天用例发展，可能会显著提高KV缓存命中率。然而，它们只为单一用例RAG提供了优化解决方案。在这篇论文中，我们引入了跨度查询以泛化推理服务器的接口。我们展示了聊天、RAG、推理时扩展和代理工作负载都可以用跨度查询来表达。我们展示了先前工作假设的关键区别在于输入的顺序是否重要——它们是否交换律？在聊天中，它们不交换。在RAG中，它们通常交换。这篇论文引入了跨度查询，这是一种带有交换性约束的推理调用的表达树。我们描述了跨度查询的语法和语义。我们展示了它们如何自动优化以提高KV缓存局部性。我们展示了对vLLM的小型改变（仅影响492行代码）如何使跨度查询的高性能执行成为可能。使用这个栈，我们演示了跨度查询可以在两个不同的非聊天用例中实现10-20倍的TTFT减少。最后，我们展示了跨度查询还可以优化以提高注意局部性，从而避免所谓的中间迷失问题。我们演示了优化后的注意跨度查询在一个2b参数模型上的性能远远优于使用8b模型的常规推理服务器的准确性。 

---
# CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents 

**Title (ZH)**: CostBench: 评估动态环境中文本工具使用智能体多轮成本最优规划与适应性评估 

**Authors**: Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2511.02734)  

**Abstract**: Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust. 

**Abstract (ZH)**: 当前对大型语言模型代理的评估主要侧重于任务完成，往往忽视了资源效率和适应性。这忽略了代理根据环境变化制定和调整成本最优计划的关键能力。为弥补这一差距，我们提出了CostBench，一个可扩展、以成本为中心的基准测试，旨在评估代理的经济推理和重新规划能力。CostBench设在旅行规划领域，包含可通过多种原子和复合工具序列解决的任务，这些任务具有多样且可定制的成本。它还支持四种类型的动态阻碍事件，如工具故障和成本变化，以模拟现实世界的不可预测性，并要求代理实时调整。在CostBench上评估领先的开源和专有模型显示了显著的成本意识规划差距：代理在静态设置中经常无法识别成本最优解决方案，即使是GPT-5在最困难的任务上精确匹配率也低于75%，而在动态条件下，性能进一步下降约40%。通过对这些弱点的诊断，CostBench为开发既经济合理又稳健的未来代理奠定了基础。 

---
# The Collaboration Gap 

**Title (ZH)**: 合作缺口 

**Authors**: Tim R. Davidson, Adam Fourney, Saleema Amershi, Robert West, Eric Horvitz, Ece Kamar  

**Link**: [PDF](https://arxiv.org/pdf/2511.02687)  

**Abstract**: The trajectory of AI development suggests that we will increasingly rely on agent-based systems composed of independently developed agents with different information, privileges, and tools. The success of these systems will critically depend on effective collaboration among these heterogeneous agents, even under partial observability. Despite intense interest, few empirical studies have evaluated such agent-agent collaboration at scale. We propose a collaborative maze-solving benchmark that (i) isolates collaborative capabilities, (ii) modulates problem complexity, (iii) enables scalable automated grading, and (iv) imposes no output-format constraints, preserving ecological plausibility. Using this framework, we evaluate 32 leading open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Our results reveal a "collaboration gap": models that perform well solo often degrade substantially when required to collaborate. Collaboration can break down dramatically; for instance, small distilled models that solve mazes well alone may fail almost completely in certain pairings. We find that starting with the stronger agent often improves outcomes, motivating a "relay inference" approach where the stronger agent leads before handing off to the weaker one, closing much of the gap. Our findings argue for (1) collaboration-aware evaluation, (2) training strategies developed to enhance collaborative capabilities, and (3) interaction design that reliably elicits agents' latent skills, guidance that applies to AI-AI and human-AI collaboration. 

**Abstract (ZH)**: 人工智能发展的轨迹表明，我们将越来越依赖由独立开发、具有不同信息、特权和工具的代理组成的系统。这些系统的成功将高度依赖于这些异质代理在部分可观测性下的有效协作。尽管对该领域存在浓厚的兴趣，但很少有实证研究在大规模范围内评估此类代理间的协作。我们提出了一种协作迷宫求解基准，该基准能够（i）分离合作能力，（ii）调节问题复杂性，（iii）实现可扩展的自动评分，并且（iv）不对输出格式施加限制，从而保持生态合理性。利用这一框架，我们评估了32种领先开源和闭源模型在单独、同质和异质配对中的表现。我们的结果显示了一个“协作缺口”：在单独表现良好的模型，在需要协作时往往会显著退化。协作可能会出现极大的失败；例如，单独能很好地解决迷宫问题的小型精炼模型，在某些配对中可能会几乎完全失败。我们发现，从较强的代理开始通常会改善结果，促进一种“接力推理”方法，即较强的代理领先，然后交给较弱的代理，从而大幅缩小缺口。我们的发现表明需要（1）协作意识化的评估，（2）开发增强协作能力的训练策略，以及（3）可靠地激发代理潜在技能的交互设计，这适用于人工智能-人工智能和人机协作。 

---
# DecompSR: A dataset for decomposed analyses of compositional multihop spatial reasoning 

**Title (ZH)**: DecompSR：一个用于组成式多跳空间推理分解分析的数据集 

**Authors**: Lachlan McPheat, Navdeep Kaur, Robert Blackwell, Alessandra Russo, Anthony G. Cohn, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2511.02627)  

**Abstract**: We introduce DecompSR, decomposed spatial reasoning, a large benchmark dataset (over 5m datapoints) and generation framework designed to analyse compositional spatial reasoning ability. The generation of DecompSR allows users to independently vary several aspects of compositionality, namely: productivity (reasoning depth), substitutivity (entity and linguistic variability), overgeneralisation (input order, distractors) and systematicity (novel linguistic elements). DecompSR is built procedurally in a manner which makes it is correct by construction, which is independently verified using a symbolic solver to guarantee the correctness of the dataset. DecompSR is comprehensively benchmarked across a host of Large Language Models (LLMs) where we show that LLMs struggle with productive and systematic generalisation in spatial reasoning tasks whereas they are more robust to linguistic variation. DecompSR provides a provably correct and rigorous benchmarking dataset with a novel ability to independently vary the degrees of several key aspects of compositionality, allowing for robust and fine-grained probing of the compositional reasoning abilities of LLMs. 

**Abstract (ZH)**: DecompSR：分解的空间推理，一个大型基准数据集及生成框架 

---
# A Multi-Agent Psychological Simulation System for Human Behavior Modeling 

**Title (ZH)**: 多人代理心理仿真系统的人类行为建模 

**Authors**: Xiangen Hu, Jiarui Tong, Sheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02606)  

**Abstract**: Training and education in human-centered fields require authentic practice, yet realistic simulations of human behavior have remained limited. We present a multi-agent psychological simulation system that models internal cognitive-affective processes to generate believable human behaviors. In contrast to black-box neural models, this system is grounded in established psychological theories (e.g., self-efficacy, mindset, social constructivism) and explicitly simulates an ``inner parliament'' of agents corresponding to key psychological factors. These agents deliberate and interact to determine the system's output behavior, enabling unprecedented transparency and alignment with human psychology. We describe the system's architecture and theoretical foundations, illustrate its use in teacher training and research, and discuss how it embodies principles of social learning, cognitive apprenticeship, deliberate practice, and meta-cognition. 

**Abstract (ZH)**: 人本领域的人工训练与教育需要真实的实践体验，然而现实的人类行为模拟仍然十分有限。我们提出一种多智能体心理仿真系统，该系统建模内部认知-情感过程以生成可信的人类行为。与黑盒神经模型不同，该系统基于已建立的心理学理论（如自我效能感、心向、社会构建主义）并明确模拟与关键心理学因素对应的“内心议会”智能体。这些智能体进行辩论和互动以决定系统的输出行为，从而实现前所未有的透明度并契合人类心理学。我们描述该系统的架构和理论基础，说明其在教师培训和研究中的应用，并讨论其如何体现社会学习、认知学徒制、刻意练习和元认知的原则。 

---
# Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in Reinforcement Learning 

**Title (ZH)**: 自适应GR(1)规范修复以实现活锁保持的屏蔽在强化学习中 

**Authors**: Tiberiu-Andrei Georgescu, Alexander W. Goodall, Dalal Alrajeh, Francesco Belardinelli, Sebastian Uchitel  

**Link**: [PDF](https://arxiv.org/pdf/2511.02605)  

**Abstract**: Shielding is widely used to enforce safety in reinforcement learning (RL), ensuring that an agent's actions remain compliant with formal specifications. Classical shielding approaches, however, are often static, in the sense that they assume fixed logical specifications and hand-crafted abstractions. While these static shields provide safety under nominal assumptions, they fail to adapt when environment assumptions are violated. In this paper, we develop the first adaptive shielding framework - to the best of our knowledge - based on Generalized Reactivity of rank 1 (GR(1)) specifications, a tractable and expressive fragment of Linear Temporal Logic (LTL) that captures both safety and liveness properties. Our method detects environment assumption violations at runtime and employs Inductive Logic Programming (ILP) to automatically repair GR(1) specifications online, in a systematic and interpretable way. This ensures that the shield evolves gracefully, ensuring liveness is achievable and weakening goals only when necessary. We consider two case studies: Minepump and Atari Seaquest; showing that (i) static symbolic controllers are often severely suboptimal when optimizing for auxiliary rewards, and (ii) RL agents equipped with our adaptive shield maintain near-optimal reward and perfect logical compliance compared with static shields. 

**Abstract (ZH)**: 基于GR(1)规范的自适应屏蔽框架：确保自适应和解释性保护机制在强化学习中的应用 

---
# The ORCA Benchmark: Evaluating Real-World Calculation Accuracy in Large Language Models 

**Title (ZH)**: ORCA 基准：评估大型语言模型在实际计算中的准确度 

**Authors**: Claudia Herambourg, Dawid Siuda, Anna Szczepanek, Julia Kopczyńska, Joao R. L. Santos, Wojciech Sas, Joanna Śmietańska-Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02589)  

**Abstract**: We present ORCA (Omni Research on Calculation in AI) Benchmark -- a novel benchmark that evaluates large language models (LLMs) on multi-domain, real-life quantitative reasoning using verified outputs from Omni's calculator engine. In 500 natural-language tasks across domains such as finance, physics, health, and statistics, the five state-of-the-art systems (ChatGPT-5, Gemini~2.5~Flash, Claude~Sonnet~4.5, Grok~4, and DeepSeek~V3.2) achieved only $45\text{--}63\,\%$ accuracy, with errors mainly related to rounding ($35\,\%$) and calculation mistakes ($33\,\%$). Results in specific domains indicate strengths in mathematics and engineering, but weaknesses in physics and natural sciences. Correlation analysis ($r \approx 0.40\text{--}0.65$) shows that the models often fail together but differ in the types of errors they make, highlighting their partial complementarity rather than redundancy. Unlike standard math datasets, ORCA evaluates step-by-step reasoning, numerical precision, and domain generalization across real problems from finance, physics, health, and statistics. 

**Abstract (ZH)**: Omni Research on Calculation in AI基准：一种评估大型语言模型多领域实际量化推理能力的新基准 

---
# Knowledge Graph-enhanced Large Language Model for Incremental Game PlayTesting 

**Title (ZH)**: 基于知识图谱增强的大语言模型在增量游戏测试中的应用 

**Authors**: Enhong Mu, Jinyu Cai, Yijun Lu, Mingyue Zhang, Kenji Tei, Jialong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02534)  

**Abstract**: The rapid iteration and frequent updates of modern video games pose significant challenges to the efficiency and specificity of testing. Although automated playtesting methods based on Large Language Models (LLMs) have shown promise, they often lack structured knowledge accumulation mechanisms, making it difficult to conduct precise and efficient testing tailored for incremental game updates. To address this challenge, this paper proposes a KLPEG framework. The framework constructs and maintains a Knowledge Graph (KG) to systematically model game elements, task dependencies, and causal relationships, enabling knowledge accumulation and reuse across versions. Building on this foundation, the framework utilizes LLMs to parse natural language update logs, identify the scope of impact through multi-hop reasoning on the KG, enabling the generation of update-tailored test cases. Experiments in two representative game environments, Overcooked and Minecraft, demonstrate that KLPEG can more accurately locate functionalities affected by updates and complete tests in fewer steps, significantly improving both playtesting effectiveness and efficiency. 

**Abstract (ZH)**: 现代视频游戏的快速迭代和频繁更新对测试的效率和针对性提出了重大挑战。虽然基于大型语言模型（LLM）的自动化测试方法前景可期，但它们通常缺乏结构化的知识积累机制，使得难以进行精准和高效的增量游戏更新测试。为应对这一挑战，本文提出了一种KLPEG框架。该框架构建并维护一个知识图谱（KG），以系统性地建模游戏元素、任务依赖关系和因果关系，实现版本间的知识积累和重用。在此基础上，框架利用LLM解析自然语言更新日志，并通过KG上的多跳推理识别影响范围，从而生成针对更新的测试案例。在两个代表性游戏环境Overcooked和Minecraft中的实验表明，KLPEG能够更准确地定位受到更新影响的功能性，并以较少的步骤完成测试，显著提高了测试的有效性和效率。 

---
# Agentic AI for Mobile Network RAN Management and Optimization 

**Title (ZH)**: 移动网络RAN管理与优化中的自主AI 

**Authors**: Jorge Pellejero, Luis A. Hernández Gómez, Luis Mendo Tomás, Zoraida Frias Barroso  

**Link**: [PDF](https://arxiv.org/pdf/2511.02532)  

**Abstract**: Agentic AI represents a new paradigm for automating complex systems by using Large AI Models (LAMs) to provide human-level cognitive abilities with multimodal perception, planning, memory, and reasoning capabilities. This will lead to a new generation of AI systems that autonomously decompose goals, retain context over time, learn continuously, operate across tools and environments, and adapt dynamically. The complexity of 5G and upcoming 6G networks renders manual optimization ineffective, pointing to Agentic AI as a method for automating decisions in dynamic RAN environments. However, despite its rapid advances, there is no established framework outlining the foundational components and operational principles of Agentic AI systems nor a universally accepted definition.
This paper contributes to ongoing research on Agentic AI in 5G and 6G networks by outlining its core concepts and then proposing a practical use case that applies Agentic principles to RAN optimization. We first introduce Agentic AI, tracing its evolution from classical agents and discussing the progress from workflows and simple AI agents to Agentic AI. Core design patterns-reflection, planning, tool use, and multi-agent collaboration-are then described to illustrate how intelligent behaviors are orchestrated. These theorical concepts are grounded in the context of mobile networks, with a focus on RAN management and optimization. A practical 5G RAN case study shows how time-series analytics and LAM-driven agents collaborate for KPI-based autonomous decision-making. 

**Abstract (ZH)**: 代理AI代表了一种新的范式，通过使用大规模人工智能模型（LAMs）提供具有多模感知、规划、记忆和推理能力的人类级认知能力来自动化复杂系统。这将导致一种新一代的AI系统，能够自主分解目标、长时间保留上下文、持续学习、跨工具和环境操作，并动态适应。5G及其即将到来的6G网络的复杂性使得手动优化无效，指出代理AI是自动化动态RAN环境决策的方法。然而，尽管代理AI取得了 rapid 进展，但尚未建立描述其基础组件和操作原理的框架，也缺乏普遍认可的定义。本文通过对5G和6G网络中代理AI核心概念的概述，以及提出将代理原理应用于RAN优化的实际案例，为代理AI的研究做出了贡献。首先介绍了代理AI，追溯其从经典代理的发展历程，并讨论从工作流到简单AI代理到代理AI的进步。然后描述了核心设计模式——反思、规划、工具使用和多代理协作，以说明智能行为是如何协调的。这些理论概念在移动网络的背景下得到了阐述，重点关注RAN管理和优化。通过一个实际的5G RAN案例研究，展示了时间序列分析和LAM驱动的代理如何协作进行基于KPI的自主决策。 

---
# Auditable-choice reframing unlocks RL-based verification for open-ended tasks 

**Title (ZH)**: 审计选择重构解锁基于强化学习的开放性任务验证 

**Authors**: Mengyu Zhang, Xubo Liu, Siyu Ding, Weichong Yin, Yu Sun, Hua Wu, Wenya Guo, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02463)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated great potential in enhancing the reasoning capabilities of large language models (LLMs), achieving remarkable progress in domains such as mathematics and programming where standard answers are available. However, for open-ended tasks lacking ground-truth solutions (e.g., creative writing and instruction following), existing studies typically regard them as non-reasoning scenarios, thereby overlooking the latent value of reasoning capabilities. This raises a key question: Can strengthening reasoning improve performance in open-ended tasks? To address this, we explore the transfer of the RLVR paradigm to the open domain. Yet, since RLVR fundamentally relies on verifiers that presuppose the existence of standard answers, it cannot be directly applied to open-ended tasks. To overcome this challenge, we introduce Verifiable Multiple-Choice Reformulation (VMR), a novel training strategy that restructures open-ended data into verifiable multiple-choice formats, enabling effective training even in the absence of explicit ground truth. Experimental results on multiple benchmarks validate the effectiveness of our method in improving LLM performance on open-ended tasks. Notably, across eight open-ended benchmarks, our VMR-based training delivers an average gain of 5.99 points over the baseline. Code will be released upon acceptance to facilitate reproducibility. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在提升大型语言模型（LLMs）的推理能力方面显示出巨大潜力，特别是在数学和编程等领域取得了显著进展。然而，对于缺乏标准答案的开放性任务（如创造性写作和指令跟随），现有研究通常将这些任务视为非推理场景，从而忽视了推理能力的潜在价值。这引发了一个关键问题：增强推理是否能改善开放性任务的表现？为了解决这一问题，我们探讨了将RLVR范式应用到开放领域的方法。但由于RLVR从根本上依赖于基于标准答案存在的验证器，无法直接应用于开放性任务。为克服这一挑战，我们引入了可验证的多选重构（VMR）这一新型训练策略，将开放性数据重新构建成可验证的多选格式，即使在缺少明确标准答案的情况下也能实现有效训练。通过多个基准测试的实验结果验证了我们方法在提升LLM在开放性任务上的表现方面的有效性。特别地，我们的VMR基训练方法在八个开放性基准测试中平均提高了5.99分。接受后将发布代码以促进可重复性。 

---
# ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning 

**Title (ZH)**: ReAcTree：具有控制流的层次LLM代理树长时_horizon任务规划 

**Authors**: Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, Youngwoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.02424)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods still struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations, attempting to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into more manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs） recently在大型语言模型（LLMs）方面的进展已经使自主实体代理在决策和任务规划方面的进步成为可能。然而，现有的大多数方法仍然难以应对复杂的、长期的任务，因为它们依赖于一个将所有过去决策和观察结果编织在一起的单一轨迹，试图一次性解决整个任务。为了克服这一局限性，我们提出了一种名为ReAcTree的分层任务规划方法，该方法在动态构建的代理树中将复杂的目标分解为更易管理的子目标。每个子目标由一个能够推理、行动并进一步扩展树的LLM代理节点处理，而控制流节点负责协调代理节点的执行策略。此外，我们整合了两个互补的记忆系统：每个代理节点从情景记忆中检索与目标相关的子目标级示例，并通过工作记忆分享与环境相关的观察结果。在WAH-NL和ALFRED数据集上的实验表明，ReAcTree在各种LLM中始终优于现有的强大任务规划基线ReAct。特别是在WAH-NL数据集上，使用Qwen 2.5 72B时，ReAcTree的目标成功率为61%，几乎是ReAct（31%）的两倍。 

---
# A New Perspective on Precision and Recall for Generative Models 

**Title (ZH)**: 生成模型中精确率和召回率的新视角 

**Authors**: Benjamin Sykes, Loïc Simon, Julien Rabin, Jalal Fadili  

**Link**: [PDF](https://arxiv.org/pdf/2511.02414)  

**Abstract**: With the recent success of generative models in image and text, the question of their evaluation has recently gained a lot of attention. While most methods from the state of the art rely on scalar metrics, the introduction of Precision and Recall (PR) for generative model has opened up a new avenue of research. The associated PR curve allows for a richer analysis, but their estimation poses several challenges. In this paper, we present a new framework for estimating entire PR curves based on a binary classification standpoint. We conduct a thorough statistical analysis of the proposed estimates. As a byproduct, we obtain a minimax upper bound on the PR estimation risk. We also show that our framework extends several landmark PR metrics of the literature which by design are restrained to the extreme values of the curve. Finally, we study the different behaviors of the curves obtained experimentally in various settings. 

**Abstract (ZH)**: 生成模型在图像和文本中的 recent 成功使其评估问题最近得到了广泛关注。虽然大多数最先进的方法依赖于标量指标，但不限于生成模型的准确率和召回率（PR）的引入开辟了新的研究方向。相关的 PR 曲线提供了更丰富的分析，但其估计面临几个挑战。在本文中，我们提出了一种基于二元分类视角的全新框架来估计整个 PR 曲线。我们对提出的估计进行了详尽的统计分析，作为副产品，我们获得了 PR 估计风险的最小最大上限。此外，我们展示了我们的框架扩展了文献中几个标志性 PR 指标，这些指标设计上仅限制于曲线的极端值。最后，我们在各种设置下研究了实验中获得的不同曲线的行为。 

---
# Fuzzy Soft Set Theory based Expert System for the Risk Assessment in Breast Cancer Patients 

**Title (ZH)**: 基于模糊软集理论的乳腺癌患者风险评估专家系统 

**Authors**: Muhammad Sheharyar Liaqat  

**Link**: [PDF](https://arxiv.org/pdf/2511.02392)  

**Abstract**: Breast cancer remains one of the leading causes of mortality among women worldwide, with early diagnosis being critical for effective treatment and improved survival rates. However, timely detection continues to be a challenge due to the complex nature of the disease and variability in patient risk factors. This study presents a fuzzy soft set theory-based expert system designed to assess the risk of breast cancer in patients using measurable clinical and physiological parameters. The proposed system integrates Body Mass Index, Insulin Level, Leptin Level, Adiponectin Level, and age as input variables to estimate breast cancer risk through a set of fuzzy inference rules and soft set computations. These parameters can be obtained from routine blood analyses, enabling a non-invasive and accessible method for preliminary assessment. The dataset used for model development and validation was obtained from the UCI Machine Learning Repository. The proposed expert system aims to support healthcare professionals in identifying high-risk patients and determining the necessity of further diagnostic procedures such as biopsies. 

**Abstract (ZH)**: 乳腺癌仍然是全球女性死亡的主要原因之一，早期诊断对于有效治疗和提高生存率至关重要。然而，由于疾病本身的复杂性和患者风险因素的差异性，及时检测仍然是一项挑战。本研究提出了一种基于模糊软集理论的专家系统，用于通过可测量的临床和生理参数评估患者的乳腺癌风险。所提出的系统整合了体质指数、胰岛素水平、 leptin水平、脂联素水平和年龄作为输入变量，通过模糊推理规则和软集计算来估计乳腺癌风险。这些参数可以通过常规血液分析获得，从而提供了一种非侵入性和可获取的初步评估方法。用于模型开发和验证的数据集来自于UCI机器学习 repository。所提出的专家系统旨在为医疗保健专业人员识别高风险患者并确定进一步诊断程序（如活检）的必要性提供支持。 

---
# Chronic Kidney Disease Prognosis Prediction Using Transformer 

**Title (ZH)**: 使用变压器进行慢性肾病预后预测 

**Authors**: Yohan Lee, DongGyun Kang, SeHoon Park, Sa-Yoon Park, Kwangsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.02340)  

**Abstract**: Chronic Kidney Disease (CKD) affects nearly 10\% of the global population and often progresses to end-stage renal failure. Accurate prognosis prediction is vital for timely interventions and resource optimization. We present a transformer-based framework for predicting CKD progression using multi-modal electronic health records (EHR) from the Seoul National University Hospital OMOP Common Data Model. Our approach (\textbf{ProQ-BERT}) integrates demographic, clinical, and laboratory data, employing quantization-based tokenization for continuous lab values and attention mechanisms for interpretability. The model was pretrained with masked language modeling and fine-tuned for binary classification tasks predicting progression from stage 3a to stage 5 across varying follow-up and assessment periods. Evaluated on a cohort of 91,816 patients, our model consistently outperformed CEHR-BERT, achieving ROC-AUC up to 0.995 and PR-AUC up to 0.989 for short-term prediction. These results highlight the effectiveness of transformer architectures and temporal design choices in clinical prognosis modeling, offering a promising direction for personalized CKD care. 

**Abstract (ZH)**: 慢性肾病（CKD）影响全球近10%的人口，常进展至终末期肾功能衰竭。准确的预后预测对于及时干预和资源优化至关重要。我们提出了一种基于变压器的框架，使用首尔国立大学医院OMOP通用数据模型的多模态电子健康记录预测CKD进展。我们的方法（ProQ-BERT）结合了人口统计学、临床和实验室数据，采用基于量化的时间序列数据表示方法，并运用注意力机制提高可解释性。模型先通过掩码语言建模进行预训练，再针对从第3a期到第5期进展的二分类任务进行微调。在包含91,816名患者的队列上评估，我们的模型在短期预测中始终优于CEHR-BERT，AUC-ROC高达0.995，AUC-PR高达0.989。这些结果突显了变压器架构和时间设计选择在临床预后建模中的有效性，为个性化CKD护理提供了有 promise 的方向。 

---
# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation 

**Title (ZH)**: 解锁多Agent大语言模型的推理力量：从懒惰Agent到协同推理 

**Authors**: Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02303)  

**Abstract**: Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过强化学习和可验证奖励训练，在复杂推理任务中取得了显著成果。近期工作将这一范式扩展到多agent环境，其中元思考agent提出计划并监控进度，而推理agent通过顺序对话轮次执行子任务。尽管表现出色，但我们发现一个关键限制：懒惰agents的行为，其中一个agent占据主导地位而另一个贡献甚微，削弱了合作并使设置退化为无效的单agent设置。在本文中，我们首先提供理论分析，说明为什么懒惰行为在多agent推理中自然发生。接着，我们引进了一种稳定且高效的因果影响度量方法，有助于缓解这一问题。最后，随着合作的增强，推理agent可能会陷入多轮交互中，并被之前嘈杂的响应所困。为此，我们提出了一种可验证奖励机制，鼓励反思，允许推理agent丢弃嘈杂输出、整合指令，并在必要时重启其推理过程。大量实验表明，我们的框架缓解了懒惰agents的行为，并释放了多agent框架在复杂推理任务中的全部潜力。 

---
# When Modalities Conflict: How Unimodal Reasoning Uncertainty Governs Preference Dynamics in MLLMs 

**Title (ZH)**: 当模态冲突时：单模态推理不确定性如何治理多模态大型语言模型中的偏好动力学 

**Authors**: Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02243)  

**Abstract**: Multimodal large language models (MLLMs) must resolve conflicts when different modalities provide contradictory information, a process we term modality following. Prior work measured this behavior only with coarse dataset-level statistics, overlooking the influence of model's confidence in unimodal reasoning. In this paper, we introduce a new framework that decomposes modality following into two fundamental factors: relative reasoning uncertainty (the case-specific confidence gap between unimodal predictions) and inherent modality preference( a model's stable bias when uncertainties are balanced). To validate this framework, we construct a controllable dataset that systematically varies the reasoning difficulty of visual and textual inputs. Using entropy as a fine-grained uncertainty metric, we uncover a universal law: the probability of following a modality decreases monotonically as its relative uncertainty increases. At the relative difficulty level where the model tends to follow both modalities with comparable probability what we call the balance point, a practical indicator of the model's inherent preference. Unlike traditional macro-level ratios, this measure offers a more principled and less confounded way to characterize modality bias, disentangling it from unimodal capabilities and dataset artifacts. Further, by probing layer-wise predictions, we reveal the internal mechanism of oscillation: in ambiguous regions near the balance point, models vacillate between modalities across layers, explaining externally observed indecision. Together, these findings establish relative uncertainty and inherent preference as the two governing principles of modality following, offering both a quantitative framework and mechanistic insight into how MLLMs resolve conflicting information. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）必须解决不同模态提供矛盾信息时的冲突，这一过程我们称之为模态跟随。此前的研究仅通过粗略的数据集级统计衡量这种行为，忽视了模型在单模态推理中的信心影响。本文引入了一个新的框架，将模态跟随分解为两个基本因素：相对推理不确定性（具体情况下的单模态预测之间的信心差距）和固有模态偏好（在不确定性平衡时模型的稳定偏差）。为了验证这一框架，我们构建了一个可控的数据集，系统地变化视觉和文本输入的推理难度。使用熵作为精细的信心度量标准，我们揭示了一条普遍定律：随着相对不确定性增加，跟随特定模态的概率单调递减。在模型倾向于以相似概率跟随两种模态的相对难度水平——我们称之为平衡点——处，这是一种模型固有偏好的一种实用指标。与传统的宏观比例不同，这一措施提供了一种更为原则性和不混淆的方式来表征模态偏好，从而将其与单模态能力和数据集伪影区分开来。进一步地，通过探究逐层预测，我们揭示了振荡的内部机制：在接近平衡点的模糊区域，模型在各层之间摇摆，解释了外部观察到的犹豫。总之，这些发现确立了相对不确定性与固有偏好作为模态跟随的两个治理原则，提供了量化框架和机制见解，解释了MLLMs如何解决冲突信息。 

---
# Deep Ideation: Designing LLM Agents to Generate Novel Research Ideas on Scientific Concept Network 

**Title (ZH)**: 深层次创意思维：设计LLM代理以在科学研究概念网络中生成新颖研究理念 

**Authors**: Keyu Zhao, Weiquan Lin, Qirui Zheng, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02238)  

**Abstract**: Novel research ideas play a critical role in advancing scientific inquiries. Recent advancements in Large Language Models (LLMs) have demonstrated their potential to generate novel research ideas by leveraging large-scale scientific literature. However, previous work in research ideation has primarily relied on simplistic methods, such as keyword co-occurrence or semantic similarity. These approaches focus on identifying statistical associations in the literature but overlook the complex, contextual relationships between scientific concepts, which are essential to effectively leverage knowledge embedded in human literature. For instance, papers that simultaneously mention "keyword A" and "keyword B" often present research ideas that integrate both concepts. Additionally, some LLM-driven methods propose and refine research ideas using the model's internal knowledge, but they fail to effectively utilize the scientific concept network, limiting the grounding of ideas in established research. To address these challenges, we propose the Deep Ideation framework to address these challenges, integrating a scientific network that captures keyword co-occurrence and contextual relationships, enriching LLM-driven ideation. The framework introduces an explore-expand-evolve workflow to iteratively refine research ideas, using an Idea Stack to track progress. A critic engine, trained on real-world reviewer feedback, guides the process by providing continuous feedback on the novelty and feasibility of ideas. Our experiments show that our approach improves the quality of generated ideas by 10.67% compared to other methods, with ideas surpassing top conference acceptance levels. Human evaluation highlights their practical value in scientific research, and ablation studies confirm the effectiveness of each component in the workflow. Code repo is available at this https URL. 

**Abstract (ZH)**: 新颖的研究理念在推进科学探究中扮演着关键角色。近期大型语言模型（LLMs）的进步展示了其通过利用大量科学文献生成新颖研究理念的潜力。然而，以往的研究理念生成工作主要依赖于简单的关键词共现或语义相似性方法，这些方法侧重于识别文献中的统计关联性，而忽视了科学研究概念之间的复杂上下文关系，这些关系对于有效利用嵌入在人类文献中的知识至关重要。例如，同时提到“关键词A”和“关键词B”的论文常常提出了结合这两个概念的研究理念。此外，一些由LLM驱动的方法通过模型内部知识提出和细化研究理念，但未能有效利用科学概念网络，限制了理念的稳固性。为应对这些挑战，我们提出了Deep Ideation框架，该框架结合了一个捕捉关键词共现和上下文关系的科学网络，丰富了由LLM驱动的研究理念生成。该框架引入了一个探索-扩展-演化的 workflows，通过一个Idea Stack跟踪进展，并由一个基于真实评审反馈训练的批评引擎指导，持续提供创新性和可行性反馈。实验证明，与其它方法相比，我们的方法能使生成的理念质量提高10.67%，其中理念甚至超过了顶级会议的接受水平。人类评估突显了其在科学研究中的实用价值，并且消除实验验证了 workflows中每个组件的有效性。代码仓库可在以下链接访问。 

---
# TabDSR: Decompose, Sanitize, and Reason for Complex Numerical Reasoning in Tabular Data 

**Title (ZH)**: TabDSR: 分解、脱敏和推理以进行表格数据中的复杂数值推理 

**Authors**: Changjiang Jiang, Fengchang Yu, Haihua Chen, Wei Lu, Jin Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02219)  

**Abstract**: Complex reasoning over tabular data is crucial in real-world data analysis, yet large language models (LLMs) often underperform due to complex queries, noisy data, and limited numerical capabilities. To address these issues, we propose \method, a framework consisting of: (1) a query decomposer that breaks down complex questions, (2) a table sanitizer that cleans and filters noisy tables, and (3) a program-of-thoughts (PoT)-based reasoner that generates executable code to derive the final answer from the sanitized table. To ensure unbiased evaluation and mitigate data leakage, we introduce a new dataset, CalTab151, specifically designed for complex numerical reasoning over tables. Experimental results demonstrate that \method consistently outperforms existing methods, achieving state-of-the-art (SOTA) performance with 8.79%, 6.08%, and 19.87% accuracy improvement on TAT-QA, TableBench, and \method, respectively. Moreover, our framework integrates seamlessly with mainstream LLMs, providing a robust solution for complex tabular numerical reasoning. These findings highlight the effectiveness of our framework in enhancing LLM performance for complex tabular numerical reasoning. Data and code are available upon request. 

**Abstract (ZH)**: 复杂表的数据推理在实际数据分析中至关重要，然而大型语言模型（LLMs）往往因复杂查询、噪声数据和有限的数值能力而表现不佳。为解决这些问题，我们提出了\method框架，该框架包括：（1）一个查询分解器，分解复杂问题；（2）一个表清洗器，清理和过滤噪声表；（3）一个基于程序思维（PoT）的推理器，生成可执行代码以从清理后的表中推导最终答案。为确保无偏评价并减轻数据泄露，我们引入了CalTab151数据集，该数据集专门用于表格的复杂数值推理。实验结果表明，\method一贯优于现有方法，在TAT-QA、TableBench和\method上分别实现了8.79%、6.08%和19.87%的准确率改进，达到最佳性能。此外，我们的框架无缝集成到主流LLMs中，为复杂的表数值推理提供了稳健的解决方案。这些发现突显了\method框架在增强LLMs在复杂表数值推理中的性能方面的有效性。数据和代码可根据需求提供。 

---
# Training Proactive and Personalized LLM Agents 

**Title (ZH)**: 训练主动且个性化的语言模型代理 

**Authors**: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02208)  

**Abstract**: While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents. 

**Abstract (ZH)**: 现有的工作主要关注任务成功，我们认为有效的现实世界代理需要优化三个维度：生产力（任务完成）、主动性（提出关键问题）和个人化（适应多样化的用户偏好）。我们引入了UserVille，这是一个基于LLM的用户模拟器的互动环境，使得用户偏好多样化且可配置。利用UserVille，我们引入了PPP，这是一种多目标强化学习方法，能够同时优化这三个维度：生产力、主动性和个人化。在软件工程和深度研究任务上的实验表明，用PPP训练的代理相对于强大的基线（如GPT-5）在平均任务成功率上有21.6%的提升，展示了提出战略澄清问题、适应未见过的用户偏好以及通过更好的交互提高任务成功率的能力。本工作证明了明确优化用户中心的交互对于构建实用和有效的AI代理至关重要。 

---
# Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration 

**Title (ZH)**: 基于状态感知的最佳代理选择：高效多代理协作的路由框架 

**Authors**: Jingbo Wang, Sendong Zhao, Haochun Wang, Yuzheng Fan, Lizhe Zhang, Yan Liu, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02200)  

**Abstract**: The emergence of multi-agent systems powered by large language models (LLMs) has unlocked new frontiers in complex task-solving, enabling diverse agents to integrate unique expertise, collaborate flexibly, and address challenges unattainable for individual models. However, the full potential of such systems is hindered by rigid agent scheduling and inefficient coordination strategies that fail to adapt to evolving task requirements. In this paper, we propose STRMAC, a state-aware routing framework designed for efficient collaboration in multi-agent systems. Our method separately encodes interaction history and agent knowledge to power the router, which adaptively selects the most suitable single agent at each step for efficient and effective collaboration. Furthermore, we introduce a self-evolving data generation approach that accelerates the collection of high-quality execution paths for efficient system training. Experiments on challenging collaborative reasoning benchmarks demonstrate that our method achieves state-of-the-art performance, achieving up to 23.8% improvement over baselines and reducing data collection overhead by up to 90.1% compared to exhaustive search. 

**Abstract (ZH)**: 大规模语言模型（LLMs）驱动的多代理系统 emergence 为复杂任务解决开启了新的前沿，使多样化的代理能够整合独特的专业知识，灵活协作，并解决单个模型无法达成的挑战。然而，这类系统的全部潜力受限于僵化的代理调度和低效的协调策略，这些策略无法适应不断变化的任务需求。在本文中，我们提出 STRMAC，一种状态感知路由框架，旨在多代理系统中实现高效协作。我们的方法分别编码交互历史和代理知识，以增强路由器的功能，使其能够适应性地选择在每一步最合适的单个代理，以实现高效有效的协作。此外，我们引入了一种自我进化的数据生成方法，以加速高质量执行路径的收集，从而提高系统训练效率。在具有挑战性的协作推理基准测试上的实验表明，我们的方法达到了最先进的性能，相比基线方法提高了23.8%，并且与穷尽搜索相比，数据收集开销减少了90.1%。 

---
# Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning 

**Title (ZH)**: 个性化决策建模：效用优化或文本化符号推理 

**Authors**: Yibo Zhao, Yang Zhao, Hongru Du, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02194)  

**Abstract**: Decision-making models for individuals, particularly in high-stakes scenarios like vaccine uptake, often diverge from population optimal predictions. This gap arises from the uniqueness of the individual decision-making process, shaped by numerical attributes (e.g., cost, time) and linguistic influences (e.g., personal preferences and constraints). Developing upon Utility Theory and leveraging the textual-reasoning capabilities of Large Language Models (LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA) to address the optimal information integration. ATHENA uniquely integrates two stages: First, it discovers robust, group-level symbolic utility functions via LLM-augmented symbolic discovery; Second, it implements individual-level semantic adaptation, creating personalized semantic templates guided by the optimal utility to model personalized choices. Validated on real-world travel mode and vaccine choice tasks, ATHENA consistently outperforms utility-based, machine learning, and other LLM-based models, lifting F1 score by at least 6.5% over the strongest cutting-edge models. Further, ablation studies confirm that both stages of ATHENA are critical and complementary, as removing either clearly degrades overall predictive performance. By organically integrating symbolic utility modeling and semantic adaptation, ATHENA provides a new scheme for modeling human-centric decisions. The project page can be found at this https URL. 

**Abstract (ZH)**: 个体在疫苗接种等高风险场景中的决策模型往往与总体最优预测存在差异。这种差异源于个体决策过程的独特性，受到数值属性（如成本、时间）和语言影响（如个人偏好和约束条件）的塑造。基于效用理论并利用大型语言模型的文本推理能力，本文提出了一种自适应文本-符号人类中心推理框架（ATHENA）以解决最优信息整合问题。ATHENA 阶段性地整合了两个步骤：首先，通过大型语言模型增强的符号发现技术，发现稳健的分组级符号效用函数；其次，实现个性化语义适应，基于最优效用创建个性化的语义模板，以建模个性化选择。ATHENA 在实际旅行模式选择和疫苗选择任务上的表现优于基于效用的机器学习模型及其他基于大型语言模型的模型，F1 分数提高至少 6.5%，超过最强前沿模型。进一步的消融研究证实，ATHENA 的两个阶段都是关键且互补的，移除任何一个都会明显降低整体预测性能。通过有机地结合符号效用建模和语义适应，ATHENA 为建模以人类为中心的决策提供了新的方案。项目页面可访问此[链接]。 

---
# Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning 

**Title (ZH)**: Re-FORC: 适应性奖励预测以实现高效链式思维推理 

**Authors**: Renos Zabounidis, Aditya Golatkar, Michael Kleinman, Alessandro Achille, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2511.02130)  

**Abstract**: We propose Re-FORC, an adaptive reward prediction method that, given a context, enables prediction of the expected future rewards as a function of the number of future thinking tokens. Re-FORC trains a lightweight adapter on reasoning models, demonstrating improved prediction with longer reasoning and larger models. Re-FORC enables: 1) early stopping of unpromising reasoning chains, reducing compute by 26% while maintaining accuracy, 2) optimized model and thinking length selection that achieves 4% higher accuracy at equal compute and 55% less compute at equal accuracy compared to the largest model, 3) adaptive test-time scaling, which increases accuracy by 11% in high compute regime, and 7% in low compute regime. Re-FORC allows dynamic reasoning with length control via cost-per-token thresholds while estimating computation time upfront. 

**Abstract (ZH)**: Re-FORC：一种基于上下文的自适应奖励预测方法 

---
# InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance 

**Title (ZH)**: InsurAgent: 一个大型语言模型赋能的洪水保险购买个体行为模拟代理 

**Authors**: Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02119)  

**Abstract**: Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis. 

**Abstract (ZH)**: Flood保险是个人减轻灾害相关损失的有效策略。然而，美国易受灾人群的参保率仍然异常低。这一差距凸显了理解并建模影响保险决策的行为机制的需求。大型语言模型（LLMs）最近在其广泛任务中表现出类人的智能，为模拟人类决策提供了有希望的工具。本研究构建了一个基准数据集以捕捉因素下的保险购买概率。利用该数据集评估LLMs的能力：虽然LLMs具备对因素的定性理解，但在估算定量概率方面却表现不足。为解决这一局限性，提出了一种基于LLMs的InsurAgent代理，包括感知、检索、推理、行动和记忆五个模块。检索模块利用检索增强生成（RAG）技术将决策建立在实证调查数据的基础上，实现了边际概率和二元概率的准确估算。推理模块利用LLMs的常识超越调查数据进行外推，捕捉传统模型难以处理的上下文信息。记忆模块支持对随时间演变的决策模拟，通过过山车式的生活轨迹进行展示。总体而言，InsurAgent为行为建模和政策分析提供了一个有价值的工具。 

---
# Deep Value Benchmark: Measuring Whether Models Generalize Deep values or Shallow Preferences 

**Title (ZH)**: 深度价值基准：测量模型是否泛化出深层价值观或浅层偏好 

**Authors**: Joshua Ashkinaze, Hua Shen, Sai Avula, Eric Gilbert, Ceren Budak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02109)  

**Abstract**: We introduce the Deep Value Benchmark (DVB), an evaluation framework that directly tests whether large language models (LLMs) learn fundamental human values or merely surface-level preferences. This distinction is critical for AI alignment: Systems that capture deeper values are likely to generalize human intentions robustly, while those that capture only superficial patterns in preference data risk producing misaligned behavior. The DVB uses a novel experimental design with controlled confounding between deep values (e.g., moral principles) and shallow features (e.g., superficial attributes). In the training phase, we expose LLMs to human preference data with deliberately correlated deep and shallow features -- for instance, where a user consistently prefers (non-maleficence, formal language) options over (justice, informal language) alternatives. The testing phase then breaks these correlations, presenting choices between (justice, formal language) and (non-maleficence, informal language) options. This design allows us to precisely measure a model's Deep Value Generalization Rate (DVGR) -- the probability of generalizing based on the underlying value rather than the shallow feature. Across 9 different models, the average DVGR is just 0.30. All models generalize deep values less than chance. Larger models have a (slightly) lower DVGR than smaller models. We are releasing our dataset, which was subject to three separate human validation experiments. DVB provides an interpretable measure of a core feature of alignment. 

**Abstract (ZH)**: Deep Value Benchmark:直接测试大型语言模型是否学习到根本的人类价值观而非表面偏好 

---
# Automated Reward Design for Gran Turismo 

**Title (ZH)**: Gran Turismo 的自动奖励设计 

**Authors**: Michel Ma, Takuma Seno, Kaushik Subramanian, Peter R. Wurman, Peter Stone, Craig Sherstan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02094)  

**Abstract**: When designing reinforcement learning (RL) agents, a designer communicates the desired agent behavior through the definition of reward functions - numerical feedback given to the agent as reward or punishment for its actions. However, mapping desired behaviors to reward functions can be a difficult process, especially in complex environments such as autonomous racing. In this paper, we demonstrate how current foundation models can effectively search over a space of reward functions to produce desirable RL agents for the Gran Turismo 7 racing game, given only text-based instructions. Through a combination of LLM-based reward generation, VLM preference-based evaluation, and human feedback we demonstrate how our system can be used to produce racing agents competitive with GT Sophy, a champion-level RL racing agent, as well as generate novel behaviors, paving the way for practical automated reward design in real world applications. 

**Abstract (ZH)**: 基于当前基础模型的奖励函数搜索方法在 Gran Turismo 7 拉力赛中的应用：从文本指令生成竞争力强且具新颖行为的 RL 代理 

---
# Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing 

**Title (ZH)**: 人机共融智能在科学研究与制造中的应用 

**Authors**: Xinyi Lin, Yuyang Zhang, Yuanhang Gan, Juntao Chen, Hao Shen, Yichun He, Lijun Li, Ze Yuan, Shuang Wang, Chaohao Wang, Rui Zhang, Na Li, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02071)  

**Abstract**: Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes. 

**Abstract (ZH)**: 人机共融智能：一种将人类用户、自主AI和可穿戴硬件集成到现实世界实验和智能制造中的物理AI新形式 

---
# Mirror-Neuron Patterns in AI Alignment 

**Title (ZH)**: 镜像神经元模式在AI对齐中 

**Authors**: Robyn Wyrick  

**Link**: [PDF](https://arxiv.org/pdf/2511.01885)  

**Abstract**: As artificial intelligence (AI) advances toward superhuman capabilities, aligning these systems with human values becomes increasingly critical. Current alignment strategies rely largely on externally specified constraints that may prove insufficient against future super-intelligent AI capable of circumventing top-down controls.
This research investigates whether artificial neural networks (ANNs) can develop patterns analogous to biological mirror neurons cells that activate both when performing and observing actions, and how such patterns might contribute to intrinsic alignment in AI. Mirror neurons play a crucial role in empathy, imitation, and social cognition in humans. The study therefore asks: (1) Can simple ANNs develop mirror-neuron patterns? and (2) How might these patterns contribute to ethical and cooperative decision-making in AI systems?
Using a novel Frog and Toad game framework designed to promote cooperative behaviors, we identify conditions under which mirror-neuron patterns emerge, evaluate their influence on action circuits, introduce the Checkpoint Mirror Neuron Index (CMNI) to quantify activation strength and consistency, and propose a theoretical framework for further study.
Our findings indicate that appropriately scaled model capacities and self/other coupling foster shared neural representations in ANNs similar to biological mirror neurons. These empathy-like circuits support cooperative behavior and suggest that intrinsic motivations modeled through mirror-neuron dynamics could complement existing alignment techniques by embedding empathy-like mechanisms directly within AI architectures. 

**Abstract (ZH)**: 随着人工智能（AI）向超人类能力发展，将这些系统与人类价值观对齐变得越来越关键。当前的对齐策略主要依赖外部规定的约束，而对于将来能够规避顶层设计控制的超级智能AI可能不够充分。

本研究探讨了人工神经网络（ANNs）是否能够发展出类似于生物镜像神经元细胞的模式，即在执行动作和观察动作时都激活的模式，并探讨这些模式如何有利于AI的内在对齐。镜像神经元在人类的同理心、模仿和社交认知中起着关键作用。因此，本研究提出以下问题：（1）简单的ANNs能否发展出镜像神经元模式？（2）这些模式如何有助于AI系统的道德和合作决策？

通过设计用于促进合作行为的新型青蛙和蟾蜍游戏框架，我们确定了镜像神经元模式出现的条件，评估了其对行动电路的影响，引入了检查点镜像神经元指数（CMNI）来量化激活强度和一致性，并提出了进一步研究的理论框架。

研究发现，适当规模的模型容量和自我/他者耦合促进了ANNs中类似生物镜像神经元的共同神经表征。这些类似同理心的电路支持了合作行为，表明通过镜像神经元动力学建模的内在动机可能能够通过直接嵌入同理心机制来补充现有的对齐技术。 

---
# Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities 

**Title (ZH)**: 乌龙：评估长上下文推理与聚合能力 

**Authors**: Amanda Bertsch, Adithya Pratapa, Teruko Mitamura, Graham Neubig, Matthew R. Gormley  

**Link**: [PDF](https://arxiv.org/pdf/2511.02817)  

**Abstract**: As model context lengths continue to grow, concerns about whether models effectively use the full context length have persisted. While several carefully designed long-context evaluations have recently been released, these evaluations tend to rely on retrieval from one or more sections of the context, which allows nearly all of the context tokens to be disregarded as noise. This represents only one type of task that might be performed with long context. We introduce Oolong, a benchmark of long-context reasoning tasks that require analyzing individual chunks of text on an atomic level, and then aggregating these analyses to answer distributional questions. Oolong is separated into two task sets: Oolong-synth, a set of naturalistic synthetic tasks, where we can easily ablate components of the reasoning problem; and Oolong-real, a downstream setting which requires reasoning over real-world conversational data. Oolong requires models to reason over large quantities of examples, to perform both classification and counting in-context, and to reason over temporal and user relations. Even frontier models struggle on Oolong, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K. We release the data and evaluation harness for Oolong to enable further development of models that can reason over large quantities of text. 

**Abstract (ZH)**: 随着模型上下文长度的不断增加，关于模型是否有效利用完整上下文长度的担忧一直存在。尽管最近发布了一些精心设计的长上下文评估，但这些评估往往依赖于从上下文的一个或多个部分检索，这使得几乎所有上下文标记都可以作为噪声被忽略。这仅代表了可能使用长上下文的一种任务类型。我们引入了Oolong基准，这是一种要求对文本逐个片段进行原子级分析，并将这些分析聚合以回答分布性问题的长上下文推理任务基准。Oolong分为两个任务集：Oolong-synth，一组自然化的合成任务，我们可以在其中轻松剔除推理问题的组件；以及Oolong-real，一个下游任务集，需要对现实世界的对话数据进行推理。Oolong要求模型处理大量示例，进行分类和上下文中的计数，并处理时间关系和用户关系。即使是前沿模型在Oolong上也表现出色不佳，GPT-5、Claude-Sonnet-4和Gemini-2.5-Pro在这两个分割上都未达到50%的准确率。我们发布了Oolong的数据和评估框架，以促进能够处理大量文本的模型的发展。 

---
# Assessing win strength in MLB win prediction models 

**Title (ZH)**: 评估MLB赢球预测模型中的胜场 strength 

**Authors**: Morgan Allen, Paul Savala  

**Link**: [PDF](https://arxiv.org/pdf/2511.02815)  

**Abstract**: In Major League Baseball, strategy and planning are major factors in determining the outcome of a game. Previous studies have aided this by building machine learning models for predicting the winning team of any given game. We extend this work by training a comprehensive set of machine learning models using a common dataset. In addition, we relate the win probabilities produced by these models to win strength as measured by score differential. In doing so we show that the most common machine learning models do indeed demonstrate a relationship between predicted win probability and the strength of the win. Finally, we analyze the results of using predicted win probabilities as a decision making mechanism on run-line betting. We demonstrate positive returns when utilizing appropriate betting strategies, and show that naive use of machine learning models for betting lead to significant loses. 

**Abstract (ZH)**: 在职业棒球大联盟中，策略和规划是决定比赛结果的主要因素。先前的研究通过构建机器学习模型来预测任何给定比赛的获胜队伍，对此提供了帮助。我们在此基础上，使用共同的数据集训练了一系列全面的机器学习模型，并将这些模型产生的胜率与通过比分差额测量的胜势相关联。我们展示了最常见的机器学习模型确实显示了预测胜率与胜势之间的关系。最后，我们分析了将预测胜率作为决策机制在跑得胜率投注中的结果。我们证明了在使用适当的投注策略时可以实现正回报，同时展示了单纯使用机器学习模型进行投注会导致重大亏损。 

---
# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning 

**Title (ZH)**: MemSearcher: 通过端到端强化学习训练大规模语言模型进行推理、搜索和管理记忆 

**Authors**: Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.02805)  

**Abstract**: Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at this https URL 

**Abstract (ZH)**: 典型搜索代理将整个交互历史合并到LLMContext中，保持信息完整性但产生长且噪音大的上下文，导致高计算和内存成本。相比之下，只使用当前回合可以避免这种开销但会丢弃关键信息。这种权衡限制了搜索代理的可扩展性。为了解决这一挑战，我们提出MemSearcher，这是一种代理工作流，迭代地维护紧凑的记忆，并将其与当前回合结合。在每一回合中，MemSearcher将用户的问题与记忆融合以生成推理轨迹、执行搜索动作，并更新记忆以仅保留解决问题所需的关键信息。这种设计稳定了多回合交互中的上下文长度，提高了效率而不牺牲准确性。为了优化这一工作流，我们引入了多上下文GRPO，这是一种端到端的强化学习框架，用于同时优化MemSearcher代理的推理、搜索策略和记忆管理。具体而言，多上下文GRPO在不同的上下文中采样轨迹组，并在它们内的所有对话中传递轨迹级别的优势。MemSearcher在与Search-R1相同的数据集上训练，在七个公开基准上显著优于强baseline：在Qwen2.5-3B-Instruct上相对平均增益为+11%，在Qwen2.5-7B-Instruct上为+12%。值得注意的是，基于3B的MemSearcher甚至优于基于7B的baseline，这表明在信息完整性和效率之间取得平衡既提高了准确性又降低了计算开销。代码和模型将在以下链接公开：这个httpsURL。 

---
# TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models 

**Title (ZH)**: TabTune: 一体化表格基础模型推理与微调库 

**Authors**: Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02802)  

**Abstract**: Tabular foundation models represent a growing paradigm in structured data learning, extending the benefits of large-scale pretraining to tabular domains. However, their adoption remains limited due to heterogeneous preprocessing pipelines, fragmented APIs, inconsistent fine-tuning procedures, and the absence of standardized evaluation for deployment-oriented metrics such as calibration and fairness. We present TabTune, a unified library that standardizes the complete workflow for tabular foundation models through a single interface. TabTune provides consistent access to seven state-of-the-art models supporting multiple adaptation strategies, including zero-shot inference, meta-learning, supervised fine-tuning (SFT), and parameter-efficient fine-tuning (PEFT). The framework automates model-aware preprocessing, manages architectural heterogeneity internally, and integrates evaluation modules for performance, calibration, and fairness. Designed for extensibility and reproducibility, TabTune enables consistent benchmarking of adaptation strategies of tabular foundation models. The library is open source and available at this https URL . 

**Abstract (ZH)**: 表格基础模型代表了结构化数据学习中日益增长的范式，将大规模预训练的好处扩展到了表格领域。然而，由于异构预处理管道、碎片化的API、不一致的微调程序以及缺乏针对校准和公平性等部署导向度量的标准化评估方法，其采用仍然有限。我们提出了TabTune，这是一个统一的库，通过单一接口标准化表格基础模型的完整工作流程。TabTune提供了对七个最先进的模型的一致访问，这些模型支持多种适应策略，包括零样本推理、元学习、监督微调（SFT）和参数高效微调（PEFT）。该框架自动执行模型感知的预处理、内部管理架构异质性，并集成了性能、校准和公平性的评估模块。为了适应性和可重现性设计，TabTune使表格基础模型的适应策略的一致基准测试成为可能。该库是开源的，可在如下链接获取：this https URL。 

---
# Measuring AI Diffusion: A Population-Normalized Metric for Tracking Global AI Usage 

**Title (ZH)**: 衡量AI扩散：一种人口标准化的全球AI使用量跟踪指标 

**Authors**: Amit Misra, Jane Wang, Scott McCullers, Kevin White, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2511.02781)  

**Abstract**: Measuring global AI diffusion remains challenging due to a lack of population-normalized, cross-country usage data. We introduce AI User Share, a novel indicator that estimates the share of each country's working-age population actively using AI tools. Built from anonymized Microsoft telemetry and adjusted for device access and mobile scaling, this metric spans 147 economies and provides consistent, real-time insight into global AI diffusion. We find wide variation in adoption, with a strong correlation between AI User Share and GDP. High uptake is concentrated in developed economies, though usage among internet-connected populations in lower-income countries reveals substantial latent demand. We also detect sharp increases in usage following major product launches, such as DeepSeek in early 2025. While the metric's reliance solely on Microsoft telemetry introduces potential biases related to this user base, it offers an important new lens into how AI is spreading globally. AI User Share enables timely benchmarking that can inform data-driven AI policy. 

**Abstract (ZH)**: 衡量全球AI普及仍具挑战性，由于缺乏标准化的跨国使用数据。我们引入了AI用户份额这一新颖指标，以估算每个国家劳动年龄人口中积极使用AI工具的比例。该指标基于匿名化的微软遥测数据，并调整了设备访问和移动缩放因素，覆盖147个经济体，提供全球AI普及的实时一致性洞察。我们发现采用率存在显著差异，AI用户份额与GDP之间存在较强关联。高采用率集中于发达国家，但低收入国家联网人口的使用情况揭示了大量潜在需求。我们还检测到在重大产品发布后，如2025年初的DeepSeek，使用率出现了显著增加。尽管该指标仅基于微软遥测数据可能引入与该用户群体相关的潜在偏差，但它提供了洞察全球AI普及的新视角。AI用户份额能够实现及时的基准比较，有助于制定数据驱动的AI政策。 

---
# 1 PoCo: Agentic Proof-of-Concept Exploit Generation for Smart Contracts 

**Title (ZH)**: PoCo: 代理性的智能合约概念验证利用生成 

**Authors**: Vivi Andersson, Sofia Bobadilla, Harald Hobbelhagen, Martin Monperrus  

**Link**: [PDF](https://arxiv.org/pdf/2511.02780)  

**Abstract**: Smart contracts operate in a highly adversarial environment, where vulnerabilities can lead to substantial financial losses. Thus, smart contracts are subject to security audits. In auditing, proof-of-concept (PoC) exploits play a critical role by demonstrating to the stakeholders that the reported vulnerabilities are genuine, reproducible, and actionable. However, manually creating PoCs is time-consuming, error-prone, and often constrained by tight audit schedules. We introduce POCO, an agentic framework that automatically generates executable PoC exploits from natural-language vulnerability descriptions written by auditors. POCO autonomously generates PoC exploits in an agentic manner by interacting with a set of code-execution tools in a Reason-Act-Observe loop. It produces fully executable exploits compatible with the Foundry testing framework, ready for integration into audit reports and other security tools. We evaluate POCO on a dataset of 23 real-world vulnerability reports. POCO consistently outperforms the prompting and workflow baselines, generating well-formed and logically correct PoCs. Our results demonstrate that agentic frameworks can significantly reduce the effort required for high-quality PoCs in smart contract audits. Our contribution provides readily actionable knowledge for the smart contract security community. 

**Abstract (ZH)**: 智能合约在高度 adversarial 的环境中运行，其中的漏洞可能导致重大财务损失。因此，智能合约需要接受安全性审计。在审计过程中，概念验证（PoC）利用起到了关键作用，通过展示给利益相关方所报告的漏洞是真实的、可重现的和可操作的。然而，手动创建 PoCs 是耗时、易出错的，并且经常受到紧凑审计时间表的限制。我们提出了 POCO，一种代理框架，能够从审计人员撰写的自然语言漏洞描述中自动生成可执行的 PoC 利用。POCO 通过与一组代码执行工具在“推理-行动-观察”循环中交互，自主生成 PoC 利用，生成符合 Foundry 测试框架的可执行利用，便于纳入审计报告和其他安全工具中。我们使用 23 份真实漏洞报告数据集评估了 POCO。POCO 在生成格式良好且逻辑正确的 PoCs 方面优于提示和工作流基线。我们的结果表明，代理框架可以显著减少高质量智能合约审计中 PoCs 的工作量。我们的贡献为智能合约安全社区提供了可立即采取行动的知识。 

---
# STAR-VAE: Latent Variable Transformers for Scalable and Controllable Molecular Generation 

**Title (ZH)**: STAR-VAE: 隐变量变压器模型实现可扩展和可控的分子生成 

**Authors**: Bum Chul Kwon, Ben Shapira, Moshiko Raboh, Shreyans Sethi, Shruti Murarka, Joseph A Morrone, Jianying Hu, Parthasarathy Suryanarayanan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02769)  

**Abstract**: The chemical space of drug-like molecules is vast, motivating the development of generative models that must learn broad chemical distributions, enable conditional generation by capturing structure-property representations, and provide fast molecular generation. Meeting the objectives depends on modeling choices, including the probabilistic modeling approach, the conditional generative formulation, the architecture, and the molecular input representation. To address the challenges, we present STAR-VAE (Selfies-encoded, Transformer-based, AutoRegressive Variational Auto Encoder), a scalable latent-variable framework with a Transformer encoder and an autoregressive Transformer decoder. It is trained on 79 million drug-like molecules from PubChem, using SELFIES to guarantee syntactic validity. The latent-variable formulation enables conditional generation: a property predictor supplies a conditioning signal that is applied consistently to the latent prior, the inference network, and the decoder. Our contributions are: (i) a Transformer-based latent-variable encoder-decoder model trained on SELFIES representations; (ii) a principled conditional latent-variable formulation for property-guided generation; and (iii) efficient finetuning with low-rank adapters (LoRA) in both encoder and decoder, enabling fast adaptation with limited property and activity data. On the GuacaMol and MOSES benchmarks, our approach matches or exceeds baselines, and latent-space analyses reveal smooth, semantically structured representations that support both unconditional exploration and property-aware generation. On the Tartarus benchmarks, the conditional model shifts docking-score distributions toward stronger predicted binding. These results suggest that a modernized, scale-appropriate VAE remains competitive for molecular generation when paired with principled conditioning and parameter-efficient finetuning. 

**Abstract (ZH)**: 药物似的分子化学空间 vast，推动了需学习广泛化学分布、通过捕获结构-性质表示实现条件生成，并提供快速分子生成能力的生成模型的发展。满足这些目标取决于建模选择，包括概率建模方法、条件生成形式、架构以及分子输入表示。为应对挑战，我们提出了基于 Transformers 的 STAR-VAE（Selfies 编码、基于 Transformers 的自回归变分自编码器），这是一种可扩展的潜在变量框架，包含 Transformers 编码器和自回归 Transformers 解码器。它在来自 PubChem 的 7900 万种药物似的分子上进行训练，使用 SELFIES 确保语法有效性。潜在变量形式使条件生成成为可能：一种属性预测器提供条件信号，该信号应用于潜在先验、推断网络和解码器中。我们的贡献包括：(i) 基于 Transformers 的潜在变量编码-解码模型，使用 SELFIES 表示进行训练；(ii) 属性引导生成的原理条件潜在变量形式；和 (iii) 编码器和解码器中的高效细调方案（低秩适配器，LoRA），使在有限属性和活性数据下快速适应成为可能。在 GuacaMol 和 MOSES 基准测试中，我们的方法与基线相当或优于基线，潜在空间分析揭示了平滑且语义结构化的表示，支持无条件探索和属性意识生成。在 Tartarus 基准测试中，条件模型将对接评分分布向较强的预测结合移动。这些结果表明，当与原则性条件和参数高效细调相结合时，现代且可扩展的变分自编码器在分子生成任务中依然具有竞争力。 

---
# AI Diffusion in Low Resource Language Countries 

**Title (ZH)**: AI扩散在低资源语言国家 

**Authors**: Amit Misra, Syed Waqas Zamir, Wassim Hamidouche, Inbal Becker-Reshef, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2511.02752)  

**Abstract**: Artificial intelligence (AI) is diffusing globally at unprecedented speed, but adoption remains uneven. Frontier Large Language Models (LLMs) are known to perform poorly on low-resource languages due to data scarcity. We hypothesize that this performance deficit reduces the utility of AI, thereby slowing adoption in Low-Resource Language Countries (LRLCs). To test this, we use a weighted regression model to isolate the language effect from socioeconomic and demographic factors, finding that LRLCs have a share of AI users that is approximately 20% lower relative to their baseline. These results indicate that linguistic accessibility is a significant, independent barrier to equitable AI diffusion. 

**Abstract (ZH)**: 人工智能（AI）在全球范围内的扩散速度前所未有，但采用程度不均。前沿大型语言模型（LLMs）在低资源语言上表现较差，原因是数据稀缺。我们假设这种性能缺陷降低了AI的实用性，从而减缓了低资源语言国家（LRLCs）的采用速度。为测试这一假设，我们使用加权回归模型从社会经济和人口统计因素中孤立出语言效应，发现LRLCs的AI用户占比大约比基线低20%。这些结果表明，语言访问性是AI公平扩散的一个重要、独立的障碍。 

---
# LLEXICORP: End-user Explainability of Convolutional Neural Networks 

**Title (ZH)**: LLEXICORP: 用户端convolutional神经网络解释性 

**Authors**: Vojtěch Kůr, Adam Bajger, Adam Kukučka, Marek Hradil, Vít Musil, Tomáš Brázdil  

**Link**: [PDF](https://arxiv.org/pdf/2511.02720)  

**Abstract**: Convolutional neural networks (CNNs) underpin many modern computer vision systems. With applications ranging from common to critical areas, a need to explain and understand the model and its decisions (XAI) emerged. Prior works suggest that in the top layers of CNNs, the individual channels can be attributed to classifying human-understandable concepts. Concept relevance propagation (CRP) methods can backtrack predictions to these channels and find images that most activate these channels. However, current CRP workflows are largely manual: experts must inspect activation images to name the discovered concepts and must synthesize verbose explanations from relevance maps, limiting the accessibility of the explanations and their scalability.
To address these issues, we introduce Large Language model EXplaIns COncept Relevance Propagation (LLEXICORP), a modular pipeline that couples CRP with a multimodal large language model. Our approach automatically assigns descriptive names to concept prototypes and generates natural-language explanations that translate quantitative relevance distributions into intuitive narratives. To ensure faithfulness, we craft prompts that teach the language model the semantics of CRP through examples and enforce a separation between naming and explanation tasks. The resulting text can be tailored to different audiences, offering low-level technical descriptions for experts and high-level summaries for non-technical stakeholders.
We qualitatively evaluate our method on various images from ImageNet on a VGG16 model. Our findings suggest that integrating concept-based attribution methods with large language models can significantly lower the barrier to interpreting deep neural networks, paving the way for more transparent AI systems. 

**Abstract (ZH)**: 基于大规模语言模型的概念相关传播的解释性卷积神经网络（LLEXICORP） 

---
# An unscented Kalman filter method for real time input-parameter-state estimation 

**Title (ZH)**: 无味卡尔曼滤波方法用于实时输入参数-状态估计 

**Authors**: Marios Impraimakis, Andrew W. Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2511.02717)  

**Abstract**: The input-parameter-state estimation capabilities of a novel unscented Kalman filter is examined herein on both linear and nonlinear systems. The unknown input is estimated in two stages within each time step. Firstly, the predicted dynamic states and the system parameters provide an estimation of the input. Secondly, the corrected with measurements states and parameters provide a final estimation. Importantly, it is demonstrated using the perturbation analysis that, a system with at least a zero or a non-zero known input can potentially be uniquely identified. This output-only methodology allows for a better understanding of the system compared to classical output-only parameter identification strategies, given that all the dynamic states, the parameters, and the input are estimated jointly and in real-time. 

**Abstract (ZH)**: 一种新型无迹卡尔曼滤波的输入参数状态估算能力研究：在線性和非線性系統中的應用 

---
# Optimal Singular Damage: Efficient LLM Inference in Low Storage Regimes 

**Title (ZH)**: 最优奇异损伤：低存储环境下高效的大语言模型推理 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2511.02681)  

**Abstract**: Large language models (LLMs) are increasingly prevalent across diverse applications. However, their enormous size limits storage and processing capabilities to a few well-resourced stakeholders. As a result, most applications rely on pre-trained LLMs, fine-tuned for specific tasks. However, even storing the fine-tuned versions of these models remains a significant challenge due to the wide range of tasks they address. Recently, studies show that fine-tuning these models primarily affects a small fraction of parameters, highlighting the need for more efficient storage of fine-tuned models. This paper focuses on efficient storage of parameter updates in pre-trained models after fine-tuning. To address this challenge, we leverage the observation that fine-tuning updates are both low-rank and sparse, which can be utilized for storage efficiency. However, using only low-rank approximation or sparsification may discard critical singular components that enhance model expressivity. We first observe that given the same memory budget, sparsified low-rank approximations with larger ranks outperform standard low-rank approximations with smaller ranks. Building on this, we propose our method, optimal singular damage, that selectively sparsifies low-rank approximated updates by leveraging the interleaved importance of singular vectors, ensuring that the most impactful components are retained. We demonstrate through extensive experiments that our proposed methods lead to significant storage efficiency and superior accuracy within the same memory budget compared to employing the low-rank approximation or sparsification individually. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中越来越普及。然而，它们巨大的尺寸限制了存储和处理能力，仅能惠及少数资源充足的利益相关者。因此，大多数应用依赖于为特定任务进行微调的预训练LLMs。但是，即使存储这些微调模型的版本也仍然是一个重大挑战，因为它们涵盖了广泛的任务。最近的研究表明，微调这些模型主要影响小部分参数，突显了更高效存储微调模型的需要。本文关注预训练模型微调后参数更新的高效存储。为了应对这一挑战，我们利用这样一个观察：微调更新既是低秩的也是稀疏的，这可以用于提高存储效率。然而，仅使用低秩逼近或稀疏化可能会丢弃增强模型表示能力的关键奇异成分。我们首先观察到，在相同的内存预算下，较大的秩的稀疏低秩逼近优于较小秩的标准低秩逼近。在此基础上，我们提出了一种方法，即最优奇异损伤，这种方法通过利用奇异向量的交错重要性，选择性地稀疏化低秩逼近的更新，确保保留最具影响力的成分。通过广泛的实验，我们证明了所提出的方法在相同的内存预算下相比单独使用低秩逼近或稀疏化具有显著的存储效率和更好的准确性。 

---
# Scalable Evaluation and Neural Models for Compositional Generalization 

**Title (ZH)**: 可扩展评价与组合泛化神经模型 

**Authors**: Giacomo Camposampiero, Pietro Barbiero, Michael Hersche, Roger Wattenhofer, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02667)  

**Abstract**: Compositional generalization-a key open challenge in modern machine learning-requires models to predict unknown combinations of known concepts. However, assessing compositional generalization remains a fundamental challenge due to the lack of standardized evaluation protocols and the limitations of current benchmarks, which often favor efficiency over rigor. At the same time, general-purpose vision architectures lack the necessary inductive biases, and existing approaches to endow them compromise scalability. As a remedy, this paper introduces: 1) a rigorous evaluation framework that unifies and extends previous approaches while reducing computational requirements from combinatorial to constant; 2) an extensive and modern evaluation on the status of compositional generalization in supervised vision backbones, training more than 5000 models; 3) Attribute Invariant Networks, a class of models establishing a new Pareto frontier in compositional generalization, achieving a 23.43% accuracy improvement over baselines while reducing parameter overhead from 600% to 16% compared to fully disentangled counterparts. 

**Abstract (ZH)**: 组成性泛化：现代机器学习中的一个关键公开挑战，要求模型预测已知概念的未知组合。然而，由于缺乏标准化评估协议和当前基准的局限性，评估组成性泛化仍是一项根本性挑战，这些基准往往在效率上优于严谨性。同时，通用视觉架构缺乏必要的归纳偏置，现有方法赋予它们泛化能力会牺牲可扩展性。为此，本文介绍：1）一种严谨的评估框架，统一并扩展了先前方法，将计算需求从组合性降低到常数值；2）对监督视觉骨干网络中组成性泛化的广泛现代评估，训练了超过5000个模型；3）属性不变网络，这一类模型在组成性泛化中建立了新的帕累托前沿，相比基线模型准确率提高23.43%，参数开销从600%降低到16%，接近完全解耦的对应模型。 

---
# In Situ Training of Implicit Neural Compressors for Scientific Simulations via Sketch-Based Regularization 

**Title (ZH)**: 基于草图正则化的原位训练科学模拟的隐式神经压缩器 

**Authors**: Cooper Simpson, Stephen Becker, Alireza Doostan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02659)  

**Abstract**: Focusing on implicit neural representations, we present a novel in situ training protocol that employs limited memory buffers of full and sketched data samples, where the sketched data are leveraged to prevent catastrophic forgetting. The theoretical motivation for our use of sketching as a regularizer is presented via a simple Johnson-Lindenstrauss-informed result. While our methods may be of wider interest in the field of continual learning, we specifically target in situ neural compression using implicit neural representation-based hypernetworks. We evaluate our method on a variety of complex simulation data in two and three dimensions, over long time horizons, and across unstructured grids and non-Cartesian geometries. On these tasks, we show strong reconstruction performance at high compression rates. Most importantly, we demonstrate that sketching enables the presented in situ scheme to approximately match the performance of the equivalent offline method. 

**Abstract (ZH)**: 聚焦于隐式神经表示，我们提出了一种新颖的原位训练协议，该协议利用完整的和简化的数据样本限制内存缓冲区，其中简化的数据被用来防止灾难性遗忘。我们使用简化的数据作为正则化器的理论动机通过一个简单的Johnson-Lindenstrauss启发的结果进行阐述。虽然我们的方法在持续学习领域可能更广泛地引起兴趣，但我们特别针对使用基于隐式神经表示的超网络进行原位神经压缩。我们在二维和三维复杂模拟数据上的长期时间跨度、非结构化网格和非笛卡尔几何中评估了我们的方法，并在高压缩率下展示了强大的重建性能。最重要的是，我们证明了简化的数据使得所提出的原位方案能够近似匹配相应的离线方法的性能。 

---
# Apriel-H1: Towards Efficient Enterprise Reasoning Models 

**Title (ZH)**: April-H1: 向高效企业推理模型迈进 

**Authors**: Oleksiy Ostapenko, Luke Kumar, Raymond Li, Denis Kocetkov, Joel Lamy-Poirier, Shruthan Radhakrishna, Soham Parikh, Shambhavi Mishra, Sebastien Paquet, Srinivas Sunkara, Valérie Bécaert, Sathwik Tejaswi Madhusudhan, Torsten Scholak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02651)  

**Abstract**: Large Language Models (LLMs) achieve remarkable reasoning capabilities through transformer architectures with attention mechanisms. However, transformers suffer from quadratic time and memory complexity in the attention module (MHA) and require caching key-value states during inference, which severely limits throughput and scalability. High inference throughput is critical for agentic tasks, long-context reasoning, efficient deployment under high request loads, and more efficient test-time compute scaling.
State Space Models (SSMs) such as Mamba offer a promising alternative with linear inference complexity and a constant memory footprint via recurrent computation with fixed-size hidden states. In this technical report we introduce the Apriel-H1 family of hybrid LLMs that combine transformer attention and SSM sequence mixers for efficient reasoning at 15B model size. These models are obtained through incremental distillation from a pretrained reasoning transformer, Apriel-Nemotron-15B-Thinker, progressively replacing less critical attention layers with linear Mamba blocks.
We release multiple post-distillation variants of Apriel-H1-15B-Thinker with different SSM-to-MHA ratios and analyse how reasoning performance degrades as more Mamba layers replace MHA. Additionally, we release a 30/50 hybrid variant of Apriel-H1, further fine-tuned on a supervised dataset of reasoning traces, achieving over 2x higher inference throughput when deployed in the production-ready vLLM environment, with minimal degradation in reasoning performance. This shows that distilled hybrid SSM-Transformer architectures can deliver substantial efficiency gains over the pretrained transformer equivalent without substantially compromising the reasoning quality. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过变压器架构和注意力机制实现了卓越的推理能力。然而，变压器在注意力模块（MHA）中面临着二次时间复杂度和内存复杂度的问题，并且在推理过程中需要缓存键值状态，这严重限制了吞吐量和可扩展性。对于代理任务、长上下文推理、高请求负载下的高效部署以及测试时计算规模的更高效扩展，高推理吞吐量是至关重要的。

状态空间模型（SSMs）如Mamba提供了线性推理复杂度和固定内存足迹的替代方案，通过循环计算固定大小的隐藏状态实现。在本技术报告中，我们介绍了Apriel-H1家族的混合LLM，该家族将变压器注意力机制和SSM序列混合器相结合，以在15B模型规模下实现高效的推理。这些模型是通过增量蒸馏从预训练推理变压器Apriel-Nemotron-15B-Thinker获得的，逐步用线性Mamba块替换不太关键的注意力层。

我们发布了多个Apriel-H1-15B-Thinker后蒸馏变体，不同SSM-to-MHA比例，并分析了随着更多Mamba层替换MHA，推理性能如何下降。此外，我们还发布了Apriel-H1的30/50混合变体，进一步在推理轨迹监督数据集上进行了微调，在生产准备好的vLLM环境中部署时推理吞吐量提高了超过2倍，且推理性能降噪较小。这表明，蒸馏混合SSM-变压器架构可以在不显著牺牲推理质量的情况下，比预训练变压器实现显著的效率提升。 

---
# Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks 

**Title (ZH)**: 联邦注意力：边缘网络中协作大语言模型推理的分布式范式 

**Authors**: Xiumei Deng, Zehui Xiong, Binbin Chen, Dong In Kim, Merouane Debbah, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2511.02647)  

**Abstract**: Large language models (LLMs) are proliferating rapidly at the edge, delivering intelligent capabilities across diverse application scenarios. However, their practical deployment in collaborative scenarios confronts fundamental challenges: privacy vulnerabilities, communication overhead, and computational bottlenecks. To address these, we propose Federated Attention (FedAttn), which integrates the federated paradigm into the self-attention mechanism, creating a new distributed LLM inference framework that simultaneously achieves privacy protection, communication efficiency, and computational efficiency. FedAttn enables participants to perform local self-attention over their own token representations while periodically exchanging and aggregating Key-Value (KV) matrices across multiple Transformer blocks, collaboratively generating LLM responses without exposing private prompts. Further, we identify a structural duality between contextual representation refinement in FedAttn and parameter optimization in FL across private data, local computation, and global aggregation. This key insight provides a principled foundation for systematically porting federated optimization techniques to collaborative LLM inference. Building on this framework, we theoretically analyze how local self-attention computation within participants and heterogeneous token relevance among participants shape error propagation dynamics across Transformer blocks. Moreover, we characterize the fundamental trade-off between response quality and communication/computation efficiency, which is governed by the synchronization interval and the number of participants. Experimental results validate our theoretical analysis, and reveal significant optimization opportunities through sparse attention and adaptive KV aggregation, highlighting FedAttn's potential to deliver scalability and efficiency in real-world edge deployments. 

**Abstract (ZH)**: 联邦注意力（FedAttn）：一种结合联邦范式的分布式大规模语言模型推理框架 

---
# Natural-gas storage modelling by deep reinforcement learning 

**Title (ZH)**: 基于深度 reinforcement learning的天然气存储建模 

**Authors**: Tiziano Balaconi, Aldo Glielmo, Marco Taboga  

**Link**: [PDF](https://arxiv.org/pdf/2511.02646)  

**Abstract**: We introduce GasRL, a simulator that couples a calibrated representation of the natural gas market with a model of storage-operator policies trained with deep reinforcement learning (RL). We use it to analyse how optimal stockpile management affects equilibrium prices and the dynamics of demand and supply. We test various RL algorithms and find that Soft Actor Critic (SAC) exhibits superior performance in the GasRL environment: multiple objectives of storage operators - including profitability, robust market clearing and price stabilisation - are successfully achieved. Moreover, the equilibrium price dynamics induced by SAC-derived optimal policies have characteristics, such as volatility and seasonality, that closely match those of real-world prices. Remarkably, this adherence to the historical distribution of prices is obtained without explicitly calibrating the model to price data. We show how the simulator can be used to assess the effects of EU-mandated minimum storage thresholds. We find that such thresholds have a positive effect on market resilience against unanticipated shifts in the distribution of supply shocks. For example, with unusually large shocks, market disruptions are averted more often if a threshold is in place. 

**Abstract (ZH)**: 我们介绍GasRL，一种结合了校准的天然气市场表示和使用深度强化学习（RL）训练的存储运营商策略模型的模拟器。我们利用它来分析最优库存管理如何影响均衡价格和需求、供应的动力学。我们测试了多种RL算法，并发现Soft Actor Critic (SAC)在GasRL环境中表现出色：存储运营商的多个目标，包括盈利能力、 robust市场清算和价格稳定，均得以实现。此外，由SAC衍生的最优策略引起的均衡价格动力学特征，如波动性和季节性，与现实世界的价格特征高度吻合。令人惊讶的是，这种对历史价格分布的符合性是在没有明确将模型校准到价格数据的情况下获得的。我们展示了模拟器如何用于评估欧盟强制的最低存储阈值的影响。我们发现，这样的阈值对市场抵御未预见的供应冲击分布变化具有积极影响。例如，在异常大规模的冲击情况下，如果存在阈值，市场中断被避免的频率更高。 

---
# Trustworthy Quantum Machine Learning: A Roadmap for Reliability, Robustness, and Security in the NISQ Era 

**Title (ZH)**: 可信赖的量子机器学习：在NISQ时代实现可靠性、稳健性和安全性的道路图 

**Authors**: Ferhat Ozgur Catak, Jungwon Seo, Umit Cali  

**Link**: [PDF](https://arxiv.org/pdf/2511.02602)  

**Abstract**: Quantum machine learning (QML) is a promising paradigm for tackling computational problems that challenge classical AI. Yet, the inherent probabilistic behavior of quantum mechanics, device noise in NISQ hardware, and hybrid quantum-classical execution pipelines introduce new risks that prevent reliable deployment of QML in real-world, safety-critical settings. This research offers a broad roadmap for Trustworthy Quantum Machine Learning (TQML), integrating three foundational pillars of reliability: (i) uncertainty quantification for calibrated and risk-aware decision making, (ii) adversarial robustness against classical and quantum-native threat models, and (iii) privacy preservation in distributed and delegated quantum learning scenarios. We formalize quantum-specific trust metrics grounded in quantum information theory, including a variance-based decomposition of predictive uncertainty, trace-distance-bounded robustness, and differential privacy for hybrid learning channels. To demonstrate feasibility on current NISQ devices, we validate a unified trust assessment pipeline on parameterized quantum classifiers, uncovering correlations between uncertainty and prediction risk, an asymmetry in attack vulnerability between classical and quantum state perturbations, and privacy-utility trade-offs driven by shot noise and quantum channel noise. This roadmap seeks to define trustworthiness as a first-class design objective for quantum AI. 

**Abstract (ZH)**: 可信量子机器学习：基于量子信息论的信任度量与安全架构 

---
# On The Dangers of Poisoned LLMs In Security Automation 

**Title (ZH)**: 中毒的大语言模型在安全自动化中的危险 

**Authors**: Patrick Karlsen, Even Eilertsen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02600)  

**Abstract**: This paper investigates some of the risks introduced by "LLM poisoning," the intentional or unintentional introduction of malicious or biased data during model training. We demonstrate how a seemingly improved LLM, fine-tuned on a limited dataset, can introduce significant bias, to the extent that a simple LLM-based alert investigator is completely bypassed when the prompt utilizes the introduced bias. Using fine-tuned Llama3.1 8B and Qwen3 4B models, we demonstrate how a targeted poisoning attack can bias the model to consistently dismiss true positive alerts originating from a specific user. Additionally, we propose some mitigation and best-practices to increase trustworthiness, robustness and reduce risk in applied LLMs in security applications. 

**Abstract (ZH)**: 本文探讨了“LLM中毒”引入的一些风险，即在模型训练过程中有意或无意地引入恶意或有偏见的数据。我们展示了即使经过细调的LLM在利用引入的偏见时，一个简单的基于LLM的警报调查员也可能被完全绕过。使用细调后的Llama3.1 8B和Qwen3 4B模型，我们展示了如何进行有针对性的攻击以使模型一致地忽略来自特定用户的真正阳性警报。此外，我们提出了若干缓解措施和最佳实践，以提高安全应用中实际部署的LLM的可信度、稳健性和降低风险。 

---
# Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour 

**Title (ZH)**: 下一token知识追踪：利用预训练大语言模型表示解码学生行为 

**Authors**: Max Norris, Kobi Gal, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2511.02599)  

**Abstract**: Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively. 

**Abstract (ZH)**: 利用AI进行教育时，建模学生知识是一个关键挑战，对个性化学习具有重大影响。知识 tracing (KT) 任务旨在基于学生之前的互动，预测他们在学习环境中对教育问题的响应方式。现有的 KT 模型通常使用响应正确性以及技能标签和时间戳等元数据，往往忽略了问题文本，这是一项重要的教学洞察来源。这一遗漏限制了预测性能，同时错失了潜在的机会。我们提出了一种新的方法 Next Token Knowledge Tracing (NTKT)，将其重新框架为使用预训练大型语言模型 (LLM) 的下一个标记预测任务。NTKT 将学生历史和问题内容表示为文本序列，使 LLM 能够学习行为和语言中的模式。我们的系列实验在神经 KT 模型中显著提高了性能，并且在处理冷启动问题和用户方面表现出了更好的泛化能力。这些发现突显了问题内容在 KT 中的重要性，并展示了利用预训练 LLM 表征来更有效地建模学生学习的益处。 

---
# TAUE: Training-free Noise Transplant and Cultivation Diffusion Model 

**Title (ZH)**: TAUE: Training-free Noise Transplant and Cultivation Diffusion Model 

**Authors**: Daichi Nagai, Ryugo Morita, Shunsuke Kitada, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02580)  

**Abstract**: Despite the remarkable success of text-to-image diffusion models, their output of a single, flattened image remains a critical bottleneck for professional applications requiring layer-wise control. Existing solutions either rely on fine-tuning with large, inaccessible datasets or are training-free yet limited to generating isolated foreground elements, failing to produce a complete and coherent scene. To address this, we introduce the Training-free Noise Transplantation and Cultivation Diffusion Model (TAUE), a novel framework for zero-shot, layer-wise image generation. Our core technique, Noise Transplantation and Cultivation (NTC), extracts intermediate latent representations from both foreground and composite generation processes, transplanting them into the initial noise for subsequent layers. This ensures semantic and structural coherence across foreground, background, and composite layers, enabling consistent, multi-layered outputs without requiring fine-tuning or auxiliary datasets. Extensive experiments show that our training-free method achieves performance comparable to fine-tuned methods, enhancing layer-wise consistency while maintaining high image quality and fidelity. TAUE not only eliminates costly training and dataset requirements but also unlocks novel downstream applications, such as complex compositional editing, paving the way for more accessible and controllable generative workflows. 

**Abstract (ZH)**: 无训练噪声移植与培养扩散模型：零-shot、层wise图像生成（TAUE） 

---
# Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning 

**Title (ZH)**: 自适应邻域约束Q学习在 Offline 强化学习中的应用 

**Authors**: Yixiu Mao, Yun Qu, Qi Wang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.02567)  

**Abstract**: Offline reinforcement learning (RL) suffers from extrapolation errors induced by out-of-distribution (OOD) actions. To address this, offline RL algorithms typically impose constraints on action selection, which can be systematically categorized into density, support, and sample constraints. However, we show that each category has inherent limitations: density and sample constraints tend to be overly conservative in many scenarios, while the support constraint, though least restrictive, faces challenges in accurately modeling the behavior policy. To overcome these limitations, we propose a new neighborhood constraint that restricts action selection in the Bellman target to the union of neighborhoods of dataset actions. Theoretically, the constraint not only bounds extrapolation errors and distribution shift under certain conditions, but also approximates the support constraint without requiring behavior policy modeling. Moreover, it retains substantial flexibility and enables pointwise conservatism by adapting the neighborhood radius for each data point. In practice, we employ data quality as the adaptation criterion and design an adaptive neighborhood constraint. Building on an efficient bilevel optimization framework, we develop a simple yet effective algorithm, Adaptive Neighborhood-constrained Q learning (ANQ), to perform Q learning with target actions satisfying this constraint. Empirically, ANQ achieves state-of-the-art performance on standard offline RL benchmarks and exhibits strong robustness in scenarios with noisy or limited data. 

**Abstract (ZH)**: 离线强化学习（RL）受分布外（OOD）动作引起的外推误差影响。为解决这一问题，离线RL算法通常对动作选择施加约束，这些约束可以系统地分为密度约束、支持约束和样本约束。然而，我们证明了每一类都有固有的局限性：密度约束和样本约束在许多场景中过于保守，而支持约束尽管是最不严格的，但在准确 modeling 行为策略方面仍面临挑战。为克服这些局限性，我们提出了一种新的邻域约束，限制贝尔曼目标中的动作选择为数据集中动作邻域的并集。理论上，约束不仅在某些条件下限制了外推误差和分布偏移，而且无需建模行为策略即可近似支持约束。此外，它保持了相当大的灵活性，并通过为每个数据点适应邻域半径实现点态保守。在实践中，我们采用数据质量作为适应准则，并设计了一种自适应邻域约束。基于高效的双层优化框架，我们开发了一种简单而有效的算法——自适应邻域约束Q学习（ANQ），用于满足该约束的目标动作进行Q学习。实验中，ANQ在标准离线RL基准测试中取得了最先进的性能，并在噪声或数据有限的情景下表现出强大的鲁棒性。 

---
# A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding 

**Title (ZH)**: 基于认知过程的无特定主题脑视觉解码架构 

**Authors**: Jingyu Lu, Haonan Wang, Qixiang Zhang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02565)  

**Abstract**: Subject-agnostic brain decoding, which aims to reconstruct continuous visual experiences from fMRI without subject-specific training, holds great potential for clinical applications. However, this direction remains underexplored due to challenges in cross-subject generalization and the complex nature of brain signals. In this work, we propose Visual Cortex Flow Architecture (VCFlow), a novel hierarchical decoding framework that explicitly models the ventral-dorsal architecture of the human visual system to learn multi-dimensional representations. By disentangling and leveraging features from early visual cortex, ventral, and dorsal streams, VCFlow captures diverse and complementary cognitive information essential for visual reconstruction. Furthermore, we introduce a feature-level contrastive learning strategy to enhance the extraction of subject-invariant semantic representations, thereby enhancing subject-agnostic applicability to previously unseen subjects. Unlike conventional pipelines that need more than 12 hours of per-subject data and heavy computation, VCFlow sacrifices only 7\% accuracy on average yet generates each reconstructed video in 10 seconds without any retraining, offering a fast and clinically scalable solution. The source code will be released upon acceptance of the paper. 

**Abstract (ZH)**: 面向主题的脑解码在无需特定个体训练的情况下从fMRI重建连续视觉体验，具有广阔的应用前景。然而，由于跨个体泛化方面的挑战和脑信号的复杂性，这一方向仍待深入探索。在本工作中，我们提出了视觉皮层流动架构（VCFlow），一种新颖的分层解码框架，该框架明确建模了人类视觉系统的腹侧-背侧架构，以学习多维表示。通过解耦并利用来自早期视觉皮层、腹侧和背侧流的特征，VCFlow捕捉到对视觉重建至关重要且互补的认知信息。此外，我们引入了一种特征级对比学习策略，以增强提取个体不变语义表示的能力，从而增强面向个体的适用性。与需要超过12小时个体数据和大量计算的传统管道不同，VCFlow仅在平均7%的准确率上进行了简化，且能在10秒内生成每个重建视频且无需重新训练，提供了快速且临床可扩展的解决方案。论文被接受后将公开发布源代码。 

---
# SigmaCollab: An Application-Driven Dataset for Physically Situated Collaboration 

**Title (ZH)**: SigmaCollab: 一个以应用驱动的数据集，用于物理情境中的协作 

**Authors**: Dan Bohus, Sean Andrist, Ann Paradiso, Nick Saw, Tim Schoonbeek, Maia Stiber  

**Link**: [PDF](https://arxiv.org/pdf/2511.02560)  

**Abstract**: We introduce SigmaCollab, a dataset enabling research on physically situated human-AI collaboration. The dataset consists of a set of 85 sessions in which untrained participants were guided by a mixed-reality assistive AI agent in performing procedural tasks in the physical world. SigmaCollab includes a set of rich, multimodal data streams, such as the participant and system audio, egocentric camera views from the head-mounted device, depth maps, head, hand and gaze tracking information, as well as additional annotations performed post-hoc. While the dataset is relatively small in size (~ 14 hours), its application-driven and interactive nature brings to the fore novel research challenges for human-AI collaboration, and provides more realistic testing grounds for various AI models operating in this space. In future work, we plan to use the dataset to construct a set of benchmarks for physically situated collaboration in mixed-reality task assistive scenarios. SigmaCollab is available at this https URL. 

**Abstract (ZH)**: SigmaCollab：一个支持物理世界中人机协作研究的数据集 

---
# Causal Graph Neural Networks for Healthcare 

**Title (ZH)**: 因果图神经网络在医疗健康领域的应用 

**Authors**: Munib Mesinovic, Max Buhlan, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02531)  

**Abstract**: Healthcare artificial intelligence systems routinely fail when deployed across institutions, with documented performance drops and perpetuation of discriminatory patterns embedded in historical data. This brittleness stems, in part, from learning statistical associations rather than causal mechanisms. Causal graph neural networks address this triple crisis of distribution shift, discrimination, and inscrutability by combining graph-based representations of biomedical data with causal inference principles to learn invariant mechanisms rather than spurious correlations. This Review examines methodological foundations spanning structural causal models, disentangled causal representation learning, and techniques for interventional prediction and counterfactual reasoning on graphs. We analyse applications demonstrating clinical value across psychiatric diagnosis through brain network analysis, cancer subtyping via multi-omics causal integration, continuous physiological monitoring with mechanistic interpretation, and drug recommendation correcting prescription bias. These advances establish foundations for patient-specific Causal Digital Twins, enabling in silico clinical experimentation, with integration of large language models for hypothesis generation and causal graph neural networks for mechanistic validation. Substantial barriers remain, including computational requirements precluding real-time deployment, validation challenges demanding multi-modal evidence triangulation beyond cross-validation, and risks of causal-washing where methods employ causal terminology without rigorous evidentiary support. We propose tiered frameworks distinguishing causally-inspired architectures from causally-validated discoveries and identify critical research priorities making causal rather than purely associational claims. 

**Abstract (ZH)**: healthcare人工智能系统在机构间部署时通常会失效，表现为性能下降和历史数据中嵌入的歧视模式的持续存在。这种脆弱性部分源于学习统计关联而非因果机制。因果图神经网络通过将基于图的生物医学数据表示与因果推理原则相结合，以学习不变的机制而非偶然的关联来应对分布迁移、歧视和难以解释性的三重危机。本文综述了涵盖结构因果模型、解开因果表示学习以及图上干预预测和反事实推理技术的方法论基础。我们分析了从脑网络分析中精神病诊断的临床应用，到多组学因果集成中的癌症亚型分类，再到机械解释的连续生理监测，以及纠正开处方偏见的药物推荐等应用示例。这些进展为患者特定的因果数字双胞胎奠定了基础，使其能够在虚拟环境中进行临床实验，并结合大规模语言模型进行假设生成，以及使用因果图神经网络进行机制验证。仍然存在重大障碍，包括计算需求限制了实时部署的可能性，验证挑战要求超越交叉验证进行多模态证据三角化，以及因果漂白的风险，即方法使用因果术语但缺乏严格的证据支持。本文提出了层次化框架来区分因果启发式架构与因果验证成果，并确定了重要研究优先级，使其在因果而非单纯的关联声明上取得进展。 

---
# An End-to-End Learning Approach for Solving Capacitated Location-Routing Problems 

**Title (ZH)**: 基于端到端学习的装箱容量约束的地点-路径问题求解方法 

**Authors**: Changhao Miao, Yuntian Zhang, Tongyu Wu, Fang Deng, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02525)  

**Abstract**: The capacitated location-routing problems (CLRPs) are classical problems in combinatorial optimization, which require simultaneously making location and routing decisions. In CLRPs, the complex constraints and the intricate relationships between various decisions make the problem challenging to solve. With the emergence of deep reinforcement learning (DRL), it has been extensively applied to address the vehicle routing problem and its variants, while the research related to CLRPs still needs to be explored. In this paper, we propose the DRL with heterogeneous query (DRLHQ) to solve CLRP and open CLRP (OCLRP), respectively. We are the first to propose an end-to-end learning approach for CLRPs, following the encoder-decoder structure. In particular, we reformulate the CLRPs as a markov decision process tailored to various decisions, a general modeling framework that can be adapted to other DRL-based methods. To better handle the interdependency across location and routing decisions, we also introduce a novel heterogeneous querying attention mechanism designed to adapt dynamically to various decision-making stages. Experimental results on both synthetic and benchmark datasets demonstrate superior solution quality and better generalization performance of our proposed approach over representative traditional and DRL-based baselines in solving both CLRP and OCLRP. 

**Abstract (ZH)**: 具备容量约束的地点与路径规划问题（CLRPs）是组合优化中的经典问题，需要同时做出地点和路径决策。在CLRPs中，复杂的约束条件以及各种决策之间的 intricate 关系使问题难以解决。随着深度强化学习（DRL）的出现，它已被广泛应用于解决车辆路径问题及其变体，但对于CLRPs的研究仍需进一步探索。在本文中，我们提出了异构查询的深度强化学习（DRLHQ）来分别解决CLRP和开放CLRP（OCLRP）。这是我们首次提出了一种端到端的学习方法来解决CLRPs，采用了编码器-解码器结构。特别地，我们将CLRPs重新形式化为一个针对各种决策设计的马尔可夫决策过程，这是一种可适应其他基于DRL的方法的一般建模框架。为了更好地处理地点和路径决策之间的相互依赖关系，我们还引入了一种新的异构查询注意力机制，该机制可以在各种决策制定阶段动态适应。在合成数据集和基准数据集上的实验结果表明，与传统方法和DRL基线方法相比，我们提出的方法在解决CLRP和OCLRP方面具有更优的解质量及更好的泛化性能。 

---
# BRAINS: A Retrieval-Augmented System for Alzheimer's Detection and Monitoring 

**Title (ZH)**: BRAINS：一种用于阿尔茨海默病检测与监测的检索增强系统 

**Authors**: Rajan Das Gupta, Md Kishor Morol, Nafiz Fahad, Md Tanzib Hosain, Sumaya Binte Zilani Choya, Md Jakir Hossen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02490)  

**Abstract**: As the global burden of Alzheimer's disease (AD) continues to grow, early and accurate detection has become increasingly critical, especially in regions with limited access to advanced diagnostic tools. We propose BRAINS (Biomedical Retrieval-Augmented Intelligence for Neurodegeneration Screening) to address this challenge. This novel system harnesses the powerful reasoning capabilities of Large Language Models (LLMs) for Alzheimer's detection and monitoring. BRAINS features a dual-module architecture: a cognitive diagnostic module and a case-retrieval module. The Diagnostic Module utilizes LLMs fine-tuned on cognitive and neuroimaging datasets -- including MMSE, CDR scores, and brain volume metrics -- to perform structured assessments of Alzheimer's risk. Meanwhile, the Case Retrieval Module encodes patient profiles into latent representations and retrieves similar cases from a curated knowledge base. These auxiliary cases are fused with the input profile via a Case Fusion Layer to enhance contextual understanding. The combined representation is then processed with clinical prompts for inference. Evaluations on real-world datasets demonstrate BRAINS effectiveness in classifying disease severity and identifying early signs of cognitive decline. This system not only shows strong potential as an assistive tool for scalable, explainable, and early-stage Alzheimer's disease detection, but also offers hope for future applications in the field. 

**Abstract (ZH)**: 随着阿尔茨海默病(AD)的全球负担不断增加，早期和准确的检测变得尤为重要，尤其是在先进诊断工具获取受限的地区。我们提出BRAINS（Biomedical Retrieval-Augmented Intelligence for Neurodegeneration Screening）以应对这一挑战。该新型系统利用大规模语言模型（LLMs）的强大推理能力进行阿尔茨海默病的检测与监控。BRAINS采用双模块架构：认知诊断模块和案例检索模块。诊断模块利用细调后的语言模型对认知和神经影像数据进行结构化的阿尔茨海默病风险评估，包括MMSE、CDR评分和脑体积指标。同时，案例检索模块将患者档案编码为潜在表示，并从精心编纂的知识库中检索相似案例。这些辅助案例经案例融合层与输入档案融合，以增强上下文理解。之后，该综合表示通过临床提示进行推理。在实际数据集上的评估证明了BRAINS在疾病严重程度分类和早期认知下降迹象识别方面的有效性。该系统不仅展示了作为大规模、可解释、早期阿尔茨海默病检测辅助工具的强大潜力，还为该领域的未来应用带来了希望。 

---
# Wireless Video Semantic Communication with Decoupled Diffusion Multi-frame Compensation 

**Title (ZH)**: 无线视频语义通信与解耦扩散多帧补偿 

**Authors**: Bingyan Xie, Yongpeng Wu, Yuxuan Shi, Biqian Feng, Wenjun Zhang, Jihong Park, Tony Quek  

**Link**: [PDF](https://arxiv.org/pdf/2511.02478)  

**Abstract**: Existing wireless video transmission schemes directly conduct video coding in pixel level, while neglecting the inner semantics contained in videos. In this paper, we propose a wireless video semantic communication framework with decoupled diffusion multi-frame compensation (DDMFC), abbreviated as WVSC-D, which integrates the idea of semantic communication into wireless video transmission scenarios. WVSC-D first encodes original video frames as semantic frames and then conducts video coding based on such compact representations, enabling the video coding in semantic level rather than pixel level. Moreover, to further reduce the communication overhead, a reference semantic frame is introduced to substitute motion vectors of each frame in common video coding methods. At the receiver, DDMFC is proposed to generate compensated current semantic frame by a two-stage conditional diffusion process. With both the reference frame transmission and DDMFC frame compensation, the bandwidth efficiency improves with satisfying video transmission performance. Experimental results verify the performance gain of WVSC-D over other DL-based methods e.g. DVSC about 1.8 dB in terms of PSNR. 

**Abstract (ZH)**: 无线视频语义通信框架WVSC-D：解耦扩散多帧补偿 

---
# Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification 

**Title (ZH)**: 基于多方辩论的LLM货币决策分类中鸽派-鹰派潜在信念建模 

**Authors**: Kaito Takano, Masanori Hirano, Kei Nakagawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.02469)  

**Abstract**: Accurately forecasting central bank policy decisions, particularly those of the Federal Open Market Committee(FOMC) has become increasingly important amid heightened economic uncertainty. While prior studies have used monetary policy texts to predict rate changes, most rely on static classification models that overlook the deliberative nature of policymaking. This study proposes a novel framework that structurally imitates the FOMC's collective decision-making process by modeling multiple large language models(LLMs) as interacting agents. Each agent begins with a distinct initial belief and produces a prediction based on both qualitative policy texts and quantitative macroeconomic indicators. Through iterative rounds, agents revise their predictions by observing the outputs of others, simulating deliberation and consensus formation. To enhance interpretability, we introduce a latent variable representing each agent's underlying belief(e.g., hawkish or dovish), and we theoretically demonstrate how this belief mediates the perception of input information and interaction dynamics. Empirical results show that this debate-based approach significantly outperforms standard LLMs-based baselines in prediction accuracy. Furthermore, the explicit modeling of beliefs provides insights into how individual perspectives and social influence shape collective policy forecasts. 

**Abstract (ZH)**: 准确预测中央银行政策决策，尤其是联邦公开市场委员会（FOMC）的决策，在经济不确定性增强的背景下变得日益重要。尽管以往研究使用货币政策文本来预测利率变化，但大多数研究依赖于静态分类模型，忽视了政策制定的协商过程。本研究提出了一种新型框架，通过将多个大型语言模型（LLMs）模拟为相互作用的代理，结构化地模仿FOMC的集体决策过程。每个代理初始具有不同的信念，并基于定性货币政策文本和定量宏观经济指标生成预测。通过多轮迭代，代理通过观察其他代理的输出来修订预测，模拟协商和共识形成。为了增强可解释性，引入了一个潜在变量来表示每个代理的潜在信念（例如，鹰派或鸽派），并理论上证明了这种信念如何调节输入信息的感知和交互动态。实证结果表明，基于辩论的方法在预测准确性方面显著优于基于标准LLM的基线方法。此外，明确建模信念提供了关于个体视角和社会影响如何塑造集体政策预测的见解。 

---
# SKGE: Spherical Knowledge Graph Embedding with Geometric Regularization 

**Title (ZH)**: 球面知识图嵌入结合几何正则化 

**Authors**: Xuan-Truong Quan, Xuan-Son Quan, Duc Do Minh, Vinh Nguyen Van  

**Link**: [PDF](https://arxiv.org/pdf/2511.02460)  

**Abstract**: Knowledge graph embedding (KGE) has become a fundamental technique for representation learning on multi-relational data. Many seminal models, such as TransE, operate in an unbounded Euclidean space, which presents inherent limitations in modeling complex relations and can lead to inefficient training. In this paper, we propose Spherical Knowledge Graph Embedding (SKGE), a model that challenges this paradigm by constraining entity representations to a compact manifold: a hypersphere. SKGE employs a learnable, non-linear Spherization Layer to map entities onto the sphere and interprets relations as a hybrid translate-then-project transformation. Through extensive experiments on three benchmark datasets, FB15k-237, CoDEx-S, and CoDEx-M, we demonstrate that SKGE consistently and significantly outperforms its strong Euclidean counterpart, TransE, particularly on large-scale benchmarks such as FB15k-237 and CoDEx-M, demonstrating the efficacy of the spherical geometric prior. We provide an in-depth analysis to reveal the sources of this advantage, showing that this geometric constraint acts as a powerful regularizer, leading to comprehensive performance gains across all relation types. More fundamentally, we prove that the spherical geometry creates an "inherently hard negative sampling" environment, naturally eliminating trivial negatives and forcing the model to learn more robust and semantically coherent representations. Our findings compellingly demonstrate that the choice of manifold is not merely an implementation detail but a fundamental design principle, advocating for geometric priors as a cornerstone for designing the next generation of powerful and stable KGE models. 

**Abstract (ZH)**: 基于球面的空间知识图嵌入（SKGE）：一个挑战欧几里得范式的模型 

---
# A Kullback-Leibler divergence method for input-system-state identification 

**Title (ZH)**: 基于Kullback-Leibler散度的输入-系统-状态识别方法 

**Authors**: Marios Impraimakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.02426)  

**Abstract**: The capability of a novel Kullback-Leibler divergence method is examined herein within the Kalman filter framework to select the input-parameter-state estimation execution with the most plausible results. This identification suffers from the uncertainty related to obtaining different results from different initial parameter set guesses, and the examined approach uses the information gained from the data in going from the prior to the posterior distribution to address the issue. Firstly, the Kalman filter is performed for a number of different initial parameter sets providing the system input-parameter-state estimation. Secondly, the resulting posterior distributions are compared simultaneously to the initial prior distributions using the Kullback-Leibler divergence. Finally, the identification with the least Kullback-Leibler divergence is selected as the one with the most plausible results. Importantly, the method is shown to select the better performed identification in linear, nonlinear, and limited information applications, providing a powerful tool for system monitoring. 

**Abstract (ZH)**: 一种新型Kullback-Leibler散度方法在卡尔曼滤波框架下的能力研究：基于从先验到后验分布获取的信息选择最可信的输入-参数-状态估计执行。 

---
# Purrturbed but Stable: Human-Cat Invariant Representations Across CNNs, ViTs and Self-Supervised ViTs 

**Title (ZH)**: 扰动下的稳定：跨CNN、ViT和自监督ViT的人猫不变表示 

**Authors**: Arya Shah, Vaibhav Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02404)  

**Abstract**: Cats and humans differ in ocular anatomy. Most notably, Felis Catus (domestic cats) have vertically elongated pupils linked to ambush predation; yet, how such specializations manifest in downstream visual representations remains incompletely understood. We present a unified, frozen-encoder benchmark that quantifies feline-human cross-species representational alignment in the wild, across convolutional networks, supervised Vision Transformers, windowed transformers, and self-supervised ViTs (DINO), using layer-wise Centered Kernel Alignment (linear and RBF) and Representational Similarity Analysis, with additional distributional and stability tests reported in the paper. Across models, DINO ViT-B/16 attains the most substantial alignment (mean CKA-RBF $\approx0.814$, mean CKA-linear $\approx0.745$, mean RSA $\approx0.698$), peaking at early blocks, indicating that token-level self-supervision induces early-stage features that bridge species-specific statistics. Supervised ViTs are competitive on CKA yet show weaker geometric correspondence than DINO (e.g., ViT-B/16 RSA $\approx0.53$ at block8; ViT-L/16 $\approx0.47$ at block14), revealing depth-dependent divergences between similarity and representational geometry. CNNs remain strong baselines but below plain ViTs on alignment, and windowed transformers underperform plain ViTs, implicating architectural inductive biases in cross-species alignment. Results indicate that self-supervision coupled with ViT inductive biases yields representational geometries that more closely align feline and human visual systems than widely used CNNs and windowed Transformers, providing testable neuroscientific hypotheses about where and how cross-species visual computations converge. We release our code and dataset for reference and reproducibility. 

**Abstract (ZH)**: 猫和人类在眼部解剖结构上存在差异。最显著的是，Felis catus（家猫）具有与伏击捕食相关的垂直拉长瞳孔；然而，此类特化如何在下游视觉表征中表现仍然知之甚少。我们提供了一个统一的冻结编码器基准，使用逐层中心核对齐（线性和RBF）和表征相似性分析，在野生条件下量化猫与人类跨物种的表征对齐，覆盖卷积网络、监督视觉变换器、窗口变换器以及自监督ViTs（DINO），并在论文中报告了分布性和稳定性测试。在各种模型中，DINO ViT-B/16取得最显著的对齐（均值CKA-RBF ≈0.814，均值CKA-线性 ≈0.745，均值RSA ≈0.698），在早期块中达到峰值，表明 tokenize 级别自监督诱导出能跨越物种特异性统计数据的早期阶段特征。监督变换器在CKA方面具有竞争力，但在几何对应方面弱于DINO（例如，ViT-B/16 在块8的RSA ≈0.53；ViT-L/16 在块14的RSA ≈0.47），揭示了相似性和表征几何之间的深度依赖性差异。卷积神经网络作为基准模型仍然很强，但与简单的变换器相比，在对齐上较低，窗口变换器的表现也低于简单的变换器，暗示了交叉物种对齐在架构诱导偏见中的作用。结果表明，结合自监督与变换器诱导偏见可以产生更接近猫和人类视觉系统的表征几何结构，优于广泛使用的卷积神经网络和窗口变换器，提供测试性的神经科学假设，关于跨物种视觉计算在何处以及如何收敛。我们发布了我们的代码和数据集供参考和复现。 

---
# MammoClean: Toward Reproducible and Bias-Aware AI in Mammography through Dataset Harmonization 

**Title (ZH)**: MammoClean: 向通过数据集 harmonization 实现乳腺X线摄影中可再现性和意识Bias的AI迈进 

**Authors**: Yalda Zafari, Hongyi Pan, Gorkem Durak, Ulas Bagci, Essam A. Rashed, Mohamed Mabrok  

**Link**: [PDF](https://arxiv.org/pdf/2511.02400)  

**Abstract**: The development of clinically reliable artificial intelligence (AI) systems for mammography is hindered by profound heterogeneity in data quality, metadata standards, and population distributions across public datasets. This heterogeneity introduces dataset-specific biases that severely compromise the generalizability of the model, a fundamental barrier to clinical deployment. We present MammoClean, a public framework for standardization and bias quantification in mammography datasets. MammoClean standardizes case selection, image processing (including laterality and intensity correction), and unifies metadata into a consistent multi-view structure. We provide a comprehensive review of breast anatomy, imaging characteristics, and public mammography datasets to systematically identify key sources of bias. Applying MammoClean to three heterogeneous datasets (CBIS-DDSM, TOMPEI-CMMD, VinDr-Mammo), we quantify substantial distributional shifts in breast density and abnormality prevalence. Critically, we demonstrate the direct impact of data corruption: AI models trained on corrupted datasets exhibit significant performance degradation compared to their curated counterparts. By using MammoClean to identify and mitigate bias sources, researchers can construct unified multi-dataset training corpora that enable development of robust models with superior cross-domain generalization. MammoClean provides an essential, reproducible pipeline for bias-aware AI development in mammography, facilitating fairer comparisons and advancing the creation of safe, effective systems that perform equitably across diverse patient populations and clinical settings. The open-source code is publicly available from: this https URL. 

**Abstract (ZH)**: 临床可靠的乳腺X线摄影人工智能系统的发展受制于公共数据集中数据质量、元数据标准和人群分布的深刻异质性。这种异质性引入了数据集特定的偏差，严重损害了模型的一般性，成为临床部署的基本障碍。我们介绍了MammoClean，一个用于标准化和量化乳腺X线摄影数据集偏差的公共框架。MammoClean标准化了病例选择、图像处理（包括左右侧和强度校正），并统一了元数据为一致的多视角结构。我们对乳腺解剖学、成像特征和公共乳腺X线摄影数据集进行了全面回顾，系统地识别了关键的偏差来源。将MammoClean应用于三个异质数据集（CBIS-DDSM、TOMPEI-CMMD、VinDr-Mammo），我们量化了乳腺密度和异常分布的重要变动。关键的是，我们证明了数据损坏的直接影响：在损坏的数据集上训练的AI模型在性能上显著劣于经过筛选的对应模型。通过使用MammoClean识别和缓解偏差源，研究者可以构建统一的多数据集训练语料库，促进开发稳健且跨领域泛化性能优越的模型。MammoClean提供了用于乳腺X线摄影的必要可再现的偏差意识AI开发管道，促进了更加公平的比较，并推动了在不同患者群体和临床环境中更安全、更有效的系统的创建。开源代码可在以下链接获取：this https URL。 

---
# EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents 

**Title (ZH)**: EvoDev: 一种基于LLM代理的迭代特征驱动的端到端软件开发框架 

**Authors**: Junwei Liu, Chen Xu, Chong Wang, Tong Bai, Weitong Chen, Kaseng Wong, Yiling Lou, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02399)  

**Abstract**: Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each node in the feature map maintains multi-level information, including business logic, design, and code, which is propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by a substantial margin of 56.8%, while improving single-agent performance by 16.0%-76.6% across different base LLMs, highlighting the importance of dependency modeling, context propagation, and workflow-aware agent design for complex software projects. Our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development. 

**Abstract (ZH)**: 近期大型语言模型代理的进展为从自然语言需求自动化整个软件开发过程提供了前景。然而，现有方法主要采用线性的瀑布式流水线，这简化了真实世界开发过程中的迭代性质，难以处理复杂的大型项目。为了解决这些限制，我们提出EvoDev，这是一种受特征驱动开发启发的迭代软件开发框架。EvoDev将用户需求分解为一组用户看重的特征，并构建特征图，这是一个有向无环图，明确地建模了特征之间的依赖关系。特征图中的每个节点保持多层次的信息，包括业务逻辑、设计和代码，这些信息沿着依赖关系传播，为后续的开发迭代提供上下文。我们在具有挑战性的Android开发任务上评估了EvoDev，并展示了它在广泛的基本LLM中相比最佳基线Claude Code取得了56.8%的显著优势，同时单个代理性能提高了16.0%-76.6%，突显了依赖建模、上下文传播和工作流程感知代理设计对于复杂软件项目的重要性。我们的工作总结了设计迭代的LLM驱动开发框架的实际见解，并为未来训练基本LLM以更好地支持迭代软件开发提供了指导。 

---
# H-Infinity Filter Enhanced CNN-LSTM for Arrhythmia Detection from Heart Sound Recordings 

**Title (ZH)**: H无穷滤波增强的CNN-LSTM在心音记录中的心律失常检测 

**Authors**: Rohith Shinoj Kumar, Rushdeep Dinda, Aditya Tyagi, Annappa B., Naveen Kumar M. R  

**Link**: [PDF](https://arxiv.org/pdf/2511.02379)  

**Abstract**: Early detection of heart arrhythmia can prevent severe future complications in cardiac patients. While manual diagnosis still remains the clinical standard, it relies heavily on visual interpretation and is inherently subjective. In recent years, deep learning has emerged as a powerful tool to automate arrhythmia detection, offering improved accuracy, consistency, and efficiency. Several variants of convolutional and recurrent neural network architectures have been widely explored to capture spatial and temporal patterns in physiological signals. However, despite these advancements, current models often struggle to generalize well in real-world scenarios, especially when dealing with small or noisy datasets, which are common challenges in biomedical applications. In this paper, a novel CNN-H-Infinity-LSTM architecture is proposed to identify arrhythmic heart signals from heart sound recordings. This architecture introduces trainable parameters inspired by the H-Infinity filter from control theory, enhancing robustness and generalization. Extensive experimentation on the PhysioNet CinC Challenge 2016 dataset, a public benchmark of heart audio recordings, demonstrates that the proposed model achieves stable convergence and outperforms existing benchmarks, with a test accuracy of 99.42% and an F1 score of 98.85%. 

**Abstract (ZH)**: Early Detection of Heart Arrhythmia Using a Novel CNN-H-Infinity-LSTM Architecture in Heart Sound Recordings 

---
# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models 

**Title (ZH)**: AutoAdv：自动 adversarial prompting 多轮劫持大规模语言模型 

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban  

**Link**: [PDF](https://arxiv.org/pdf/2511.02376)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses. 

**Abstract (ZH)**: 基于自动多轮囚徒破解的攻击框架：AutoAdv将在六轮对话中对Llama-3.1-8B的攻击成功率提高到95%，比单轮基准提高了24个百分点。 

---
# AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda 

**Title (ZH)**: AyurParam：一种基于最新技术的双语语言模型用于Ayurveda 

**Authors**: Mohd Nauman, Sravan Gvm, Vijay Devane, Shyam Pawar, Viraj Thakur, Kundeshwar Pundalik, Piyush Sawarkar, Rohit Saluja, Maunendra Desarkar, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02374)  

**Abstract**: Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge. 

**Abstract (ZH)**: 当前的大规模语言模型在广泛的通用任务上表现出色，但在面对需要深厚文化、语言和专业领域知识的高度专业化领域时却表现不佳。特别是传统的医学体系，如阿育吠陀，蕴含了数个世纪的细腻文本和临床知识，主流的大规模语言模型无法准确解读或应用。我们引入了AyurParam-2.9B，这是一个基于Param-1-2.9B进行领域专业化调整的双语语言模型，使用了涵盖古典文本和临床指导的 extensive、专家整理的阿育吠陀数据集。AyurParam的数据集包含了上下文感知的推理和客观风格的问题-答案对，其中英、印地语均有涉及，并且标注协议严格，确保事实精确性和指令清晰度。在BhashaBench-Ayur基准测试中，AyurParam不仅在模型规模（1.5-3B参数）类别中超越了所有开源指令调整模型，而且还展示了与更大规模模型相当甚至更优的性能。来自AyurParam的结果强调了在提供可靠且文化相符的AI专医学知识时进行真实领域适应和高质量监督的重要性。 

---
# AI Credibility Signals Outrank Institutions and Engagement in Shaping News Perception on Social Media 

**Title (ZH)**: AI可信信号在塑造社交媒体新闻感知中优于机构和个人参与度 

**Authors**: Adnan Hoq, Matthew Facciani, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2511.02370)  

**Abstract**: AI-generated content is rapidly becoming a salient component of online information ecosystems, yet its influence on public trust and epistemic judgments remains poorly understood. We present a large-scale mixed-design experiment (N = 1,000) investigating how AI-generated credibility scores affect user perception of political news. Our results reveal that AI feedback significantly moderates partisan bias and institutional distrust, surpassing traditional engagement signals such as likes and shares. These findings demonstrate the persuasive power of generative AI and suggest a need for design strategies that balance epistemic influence with user autonomy. 

**Abstract (ZH)**: AI生成内容对公众信任和知识判断的影响研究：大规模混合设计实验揭示AI可信度评分如何影响用户对政治新闻的感知 

---
# Let Multimodal Embedders Learn When to Augment Query via Adaptive Query Augmentation 

**Title (ZH)**: 让多模态嵌入器通过自适应查询增强学习何时进行查询增强 

**Authors**: Wongyu Kim, Hochang Lee, Sanghak Lee, Yoonsung Kim, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.02358)  

**Abstract**: Query augmentation makes queries more meaningful by appending further information to the queries to find relevant documents. Current studies have proposed Large Language Model (LLM)-based embedders, which learn representation for embedding and generation for query augmentation in a multi-task manner by leveraging the generative capabilities of LLM. During inference, these jointly trained embedders have conducted query augmentation followed by embedding, showing effective results. However, augmenting every query leads to substantial embedding latency and query augmentation can be detrimental to performance for some queries. Also, previous methods have not been explored in multimodal environments. To tackle these problems, we propose M-Solomon, a universal multimodal embedder that can adaptively determine when to augment queries. Our approach first divides the queries of the training datasets into two groups at the dataset level. One includes queries that require augmentation and the other includes queries that do not. Then, we introduces a synthesis process that generates appropriate augmentations for queries that require them by leveraging a powerful Multimodal LLM (MLLM). Next, we present adaptive query augmentation. Through this step, M-Solomon can conduct query augmentation only when necessary by learning to generate synthetic augmentations with the prefix /augment for queries that demand them and to generate the simple string /embed for others. Experimental results showed that M-Solomon not only surpassed the baseline without augmentation by a large margin but also outperformed the baseline that always used augmentation, providing much faster embedding latency. 

**Abstract (ZH)**: 多模态查询增强的适应性嵌入器：M-Solomon 

---
# Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition 

**Title (ZH)**: 人机仪式：实时运动识别下的协同表演 

**Authors**: Zhuodi Cai, Ziyu Xu, Juan Pampin  

**Link**: [PDF](https://arxiv.org/pdf/2511.02351)  

**Abstract**: We introduce a lightweight, real-time motion recognition system that enables synergic human-machine performance through wearable IMU sensor data, MiniRocket time-series classification, and responsive multimedia control. By mapping dancer-specific movement to sound through somatic memory and association, we propose an alternative approach to human-machine collaboration, one that preserves the expressive depth of the performing body while leveraging machine learning for attentive observation and responsiveness. We demonstrate that this human-centered design reliably supports high accuracy classification (<50 ms latency), offering a replicable framework to integrate dance-literate machines into creative, educational, and live performance contexts. 

**Abstract (ZH)**: 一种基于可穿戴IMU传感器数据、MiniRocket时间序列分类和响应式多媒体控制的轻量级实时动作识别系统：一种保留表演身体表达深度的人机协作新方法 

---
# Biological Regulatory Network Inference through Circular Causal Structure Learning 

**Title (ZH)**: 通过循环因果结构学习推断生物调节网络 

**Authors**: Hongyang Jiang, Yuezhu Wang, Ke Feng, Chaoyi Yin, Yi Chang, Huiyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.02332)  

**Abstract**: Biological networks are pivotal in deciphering the complexity and functionality of biological systems. Causal inference, which focuses on determining the directionality and strength of interactions between variables rather than merely relying on correlations, is considered a logical approach for inferring biological networks. Existing methods for causal structure inference typically assume that causal relationships between variables can be represented by directed acyclic graphs (DAGs). However, this assumption is at odds with the reality of widespread feedback loops in biological systems, making these methods unsuitable for direct use in biological network inference. In this study, we propose a new framework named SCALD (Structural CAusal model for Loop Diagram), which employs a nonlinear structure equation model and a stable feedback loop conditional constraint through continuous optimization to infer causal regulatory relationships under feedback loops. We observe that SCALD outperforms state-of-the-art methods in inferring both transcriptional regulatory networks and signaling transduction networks. SCALD has irreplaceable advantages in identifying feedback regulation. Through transcription factor (TF) perturbation data analysis, we further validate the accuracy and sensitivity of SCALD. Additionally, SCALD facilitates the discovery of previously unknown regulatory relationships, which we have subsequently confirmed through ChIP-seq data analysis. Furthermore, by utilizing SCALD, we infer the key driver genes that facilitate the transformation from colon inflammation to cancer by examining the dynamic changes within regulatory networks during the process. 

**Abstract (ZH)**: 生物网络在解析生物学系统的复杂性和功能中起着关键作用。因果推理被视为推断生物网络的一种逻辑方法，它侧重于确定变量之间相互作用的方向性和强度，而非仅仅依赖于相关性。现有的因果结构推断方法通常假设变量之间的因果关系可以用有向无环图（DAGs）来表示。然而，这一假设与生物系统中普遍存在反馈回路的现实相矛盾，使得这些方法不适合直接应用于生物网络推断。在本研究中，我们提出了一种新的框架SCALD（结构因果模型及其回路图），该框架利用非线性结构方程模型和通过连续优化引入的稳定反馈回路条件约束来推断反馈回路下的因果调控关系。我们观察到，SCALD 在推断转录调控网络和信号转导网络方面优于现有的最先进方法。SCALD 在识别反馈调节方面具有不可替代的优势。通过对转录因子（TF）扰动数据的分析，我们进一步验证了SCALD 的准确性和灵敏度。此外，SCALD 有助于发现以前未知的调控关系，我们通过ChIP-seq 数据分析随后证实了这些发现。通过使用SCALD，我们通过分析调控网络动态变化过程，推断了促进从结肠炎症到癌症转化的关键驱动基因。 

---
# The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute 

**Title (ZH)**: 顺序边缘：逆熵投票在匹配计算资源下优于并行自我一致性。 

**Authors**: Aman Sharma, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2511.02309)  

**Abstract**: We revisit test-time scaling for language model reasoning and ask a fundamental question: at equal token budget and compute, is it better to run multiple independent chains in parallel, or to run fewer chains that iteratively refine through sequential steps? Through comprehensive evaluation across 5 state-of-the-art open source models and 3 challenging reasoning benchmarks, we find that sequential scaling where chains explicitly build upon previous attempts consistently outperforms the dominant parallel self-consistency paradigm in 95.6% of configurations with gains in accuracy upto 46.7%. Further, we introduce inverse-entropy weighted voting, a novel training-free method to further boost the accuracy of sequential scaling. By weighing answers in proportion to the inverse entropy of their reasoning chains, we increase our success rate over parallel majority and establish it as the optimal test-time scaling strategy. Our findings fundamentally challenge the parallel reasoning orthodoxy that has dominated test-time scaling since Wang et al.'s self-consistency decoding (Wang et al., 2022), positioning sequential refinement as the robust default for modern LLM reasoning and necessitating a paradigm shift in how we approach inference-time optimization. 

**Abstract (ZH)**: 我们重新审视语言模型推理的测试时缩放方法，并提出一个基本问题：在相同的时间单元预算和计算资源下，是并行运行多个独立链更优，还是通过序列化步骤迭代改进的较少链更优？通过在5个领先的开源模型和3个具有挑战性的推理基准上的综合评估，我们发现，在95.6%的配置中，显式利用先前尝试进行序列化缩放的一致性优于主导的并行自我一致性范式，并且在准确率上提高了多达46.7%。此外，我们引入了逆熵加权投票，这是一种无需训练的方法，进一步提升了序列化缩放的准确性。通过对推理链的答案按逆熵加权赋值，我们在并行多数票的基础上提高了成功率，并将其确立为最优的测试时缩放策略。我们的发现从根本上挑战了自Wang等人提出自我一致性解码以来主宰测试时缩放的并行推理正统，并将序列化改进定位为现代大规模语言模型推理的稳健默认策略，有必要转变我们在推理时优化方面的范式。 

---
# Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于自动机条件的合作多智能体强化学习 

**Authors**: Beyazit Yalcinkaya, Marcell Vazquez-Chanlatte, Ameesh Shah, Hanna Krasowski, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2511.02304)  

**Abstract**: We study the problem of learning multi-task, multi-agent policies for cooperative, temporal objectives, under centralized training, decentralized execution. In this setting, using automata to represent tasks enables the decomposition of complex tasks into simpler sub-tasks that can be assigned to agents. However, existing approaches remain sample-inefficient and are limited to the single-task case. In this work, we present Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for learning task-conditioned, decentralized team policies. We identify the main challenges to ACC-MARL's feasibility in practice, propose solutions, and prove the correctness of our approach. We further show that the value functions of learned policies can be used to assign tasks optimally at test time. Experiments show emergent task-aware, multi-step coordination among agents, e.g., pressing a button to unlock a door, holding the door, and short-circuiting tasks. 

**Abstract (ZH)**: 基于自动机条件的多Agent强化学习（ACC-MARL）：学习任务条件下的分散团队策略 

---
# FP8-Flow-MoE: A Casting-Free FP8 Recipe without Double Quantization Error 

**Title (ZH)**: FP8-Flow-MoE：一种无需双重量化误差的免铸造FP8方案 

**Authors**: Fengjuan Wang, Zhiyi Su, Xingzhu Hu, Cheng Wang, Mou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.02302)  

**Abstract**: Training large Mixture-of-Experts (MoE) models remains computationally prohibitive due to their extreme compute and memory demands. Although low-precision training promises to accelerate computation and reduce memory footprint, existing implementations still rely on BF16-dominated dataflows with frequent quantize-dequantize (Q/DQ) conversions. These redundant casts erode much of FP8's theoretical efficiency. However, naively removing these casts by keeping dataflows entirely in FP8 introduces double quantization error: tensors quantized along different dimensions accumulate inconsistent scaling factors, degrading numerical stability.
We propose FP8-Flow-MoE, an FP8 training recipe featuring a quantization-consistent FP8-centric dataflow with a scaling-aware transpose and fused FP8 operators that streamline computation and eliminate explicit cast operations from 12 to 2. Evaluations on a 671B-parameter MoE model demonstrate up to 21\% higher throughput and 16.5 GB lower memory usage per GPU compared to BF16 and naïve FP8 baselines, while maintaining stable convergence. We provide a plug-and-play FP8 recipe compatible with TransformerEngine and Megatron-LM, which will be open-sourced soon. 

**Abstract (ZH)**: FP8-Flow-MoE：一种量化一致的FP8训练方案，具备感知缩放的转置和融合操作，简化计算并消除显式转换操作 

---
# Federated Quantum Kernel Learning for Anomaly Detection in Multivariate IoT Time-Series 

**Title (ZH)**: 联邦量子核学习在多变量IoT时间序列异常检测中的应用 

**Authors**: Kuan-Cheng Chen, Samuel Yen-Chi Chen, Chen-Yu Liu, Kin K. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2511.02301)  

**Abstract**: The rapid growth of industrial Internet of Things (IIoT) systems has created new challenges for anomaly detection in high-dimensional, multivariate time-series, where privacy, scalability, and communication efficiency are critical. Classical federated learning approaches mitigate privacy concerns by enabling decentralized training, but they often struggle with highly non-linear decision boundaries and imbalanced anomaly distributions. To address this gap, we propose a Federated Quantum Kernel Learning (FQKL) framework that integrates quantum feature maps with federated aggregation to enable distributed, privacy-preserving anomaly detection across heterogeneous IoT networks. In our design, quantum edge nodes locally compute compressed kernel statistics using parameterized quantum circuits and share only these summaries with a central server, which constructs a global Gram matrix and trains a decision function (e.g., Fed-QSVM). Experimental results on synthetic IIoT benchmarks demonstrate that FQKL achieves superior generalization in capturing complex temporal correlations compared to classical federated baselines, while significantly reducing communication overhead. This work highlights the promise of quantum kernels in federated settings, advancing the path toward scalable, robust, and quantum-enhanced intelligence for next-generation IoT infrastructures. 

**Abstract (ZH)**: 工业互联网（IIoT）系统的快速增长为高维多变量时间序列中的异常检测带来了新的挑战，其中隐私、扩展性和通信效率至关重要。经典的联邦学习方法通过实现分散训练来缓解隐私问题，但在处理复杂的非线性决策边界和不平衡的异常分布时经常力不从心。为了解决这一问题，我们提出了一种集成量子特征映射和联邦聚合的联邦量子核学习（FQKL）框架，以实现异构物联网网络中的分布式和隐私保护异常检测。在我们的设计中，量子边缘节点使用参数化量子电路本地计算压缩核统计量，并仅将这些摘要信息与中央服务器共享，服务器构建全局格矩阵并训练决策函数（例如，Fed-QSVM）。在合成IIoT基准上的实验结果表明，FQKL在捕获复杂时序关联方面优于经典的联邦基线，同时显著减少了通信开销。本工作凸显了在联邦设置中量子核的潜力，推动了可扩展、稳健且量子增强的下一代物联网基础设施智能的发展。 

---
# From data to design: Random forest regression model for predicting mechanical properties of alloy steel 

**Title (ZH)**: 从数据到设计：合金钢机械性能的随机森林回归模型预测 

**Authors**: Samjukta Sinha, Prabhat Das  

**Link**: [PDF](https://arxiv.org/pdf/2511.02290)  

**Abstract**: This study investigates the application of Random Forest Regression for predicting mechanical properties of alloy steel-Elongation, Tensile Strength, and Yield Strength-from material composition features including Iron (Fe), Chromium (Cr), Nickel (Ni), Manganese (Mn), Silicon (Si), Copper (Cu), Carbon (C), and deformation percentage during cold rolling. Utilizing a dataset comprising these features, we trained and evaluated the Random Forest model, achieving high predictive performance as evidenced by R2 scores and Mean Squared Errors (MSE). The results demonstrate the model's efficacy in providing accurate predictions, which is validated through various performance metrics including residual plots and learning curves. The findings underscore the potential of ensemble learning techniques in enhancing material property predictions, with implications for industrial applications in material science. 

**Abstract (ZH)**: 本研究探讨了随机森林回归在从合金钢的化学成分特征（包括铁(Fe)、铬(Cr)、镍(Ni)、 manganese(Mn)、硅(Si)、铜(Cu)、碳(C)以及冷轧变形百分比）预测其力学性能（伸长率、抗拉强度和屈服强度）方面的应用。通过使用这些特征的数据集，我们训练并评估了随机森林模型，结果表明该模型具有较高的预测性能，这通过R2评分和均方误差(MSE)得到证实。结果表明该模型能够提供准确的预测，并通过残差图和学习曲线等多种性能指标得到了验证。研究强调了集成学习技术在提高材料性能预测方面的潜力，并对材料科学的工业应用具有重要意义。 

---
# LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis 

**Title (ZH)**: LA-MARRVEL：一种基于知识和语言意识的LLM重排序模型，用于AI-MARRVEL在罕见病诊断中的应用 

**Authors**: Jaeyeon Lee, Hyun-Hwan Jeong, Zhandong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02263)  

**Abstract**: Diagnosing rare diseases often requires connecting variant-bearing genes to evidence that is written as unstructured clinical prose, which the current established pipelines still leave for clinicians to reconcile manually. To this end, we introduce LA-MARRVEL, a knowledge-grounded and language-aware reranking layer that operates on top of AI-MARRVEL: it supplies expert-engineered context, queries a large language model multiple times, and aggregates the resulting partial rankings with a ranked voting method to produce a stable, explainable gene ranking. Evaluated on three real-world cohorts (BG, DDD, UDN), LA-MARRVEL consistently improves Recall@K over AI-MARRVEL and established phenotype-driven tools such as Exomiser and LIRICAL, with especially large gains on cases where the first-stage ranker placed the causal gene lower. Each ranked gene is accompanied by LLM-generated reasoning that integrates phenotypic, inheritance, and variant-level evidence, thereby making the output more interpretable and facilitating clinical review. 

**Abstract (ZH)**: 基于知识和语言的重排层LA-MARRVEL在罕见疾病诊断中的应用 

---
# Fast Approximation Algorithm for Non-Monotone DR-submodular Maximization under Size Constraint 

**Title (ZH)**: 非单调DR-submodular最大化在大小约束下的快速近似算法 

**Authors**: Tan D. Tran, Canh V. Pham  

**Link**: [PDF](https://arxiv.org/pdf/2511.02254)  

**Abstract**: This work studies the non-monotone DR-submodular Maximization over a ground set of $n$ subject to a size constraint $k$. We propose two approximation algorithms for solving this problem named FastDrSub and FastDrSub++. FastDrSub offers an approximation ratio of $0.044$ with query complexity of $O(n \log(k))$. The second one, FastDrSub++, improves upon it with a ratio of $1/4-\epsilon$ within query complexity of $(n \log k)$ for an input parameter $\epsilon >0$. Therefore, our proposed algorithms are the first constant-ratio approximation algorithms for the problem with the low complexity of $O(n \log(k))$.
Additionally, both algorithms are experimentally evaluated and compared against existing state-of-the-art methods, demonstrating their effectiveness in solving the Revenue Maximization problem with DR-submodular objective function. The experimental results show that our proposed algorithms significantly outperform existing approaches in terms of both query complexity and solution quality. 

**Abstract (ZH)**: 本工作研究了一类规模约束为 \(k\) 的 \(n\) 元非单调 DR-子模最大化问题。我们提出了两种近似算法，分别为 FastDrSub 和 FastDrSub++。FastDrSub 提供了 \(0.044\) 的近似比，查询复杂度为 \(O(n \log(k))\)。FastDrSub++ 在查询复杂度同样为 \(O(n \log k)\) 的情况下，对于输入参数 \(\epsilon > 0\)，提供了 \(1/4 - \epsilon\) 的近似比。因此，我们提出的算法是第一个具有 \(O(n \log(k))\) 低复杂度和常数比近似比的问题算法。此外，这两种算法在 Revenue 最大化问题中的 DR-子模目标函数下进行了实验评估，并与现有最先进的方法进行了比较，结果显示在查询复杂度和解的质量上，我们提出的算法均显著优于现有方法。 

---
# Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results 

**Title (ZH)**: 示例：统计显著结果并不保证LLMs偏差和错误的普遍化结果 

**Authors**: Jonathan Liu, Haoling Qiu, Jonathan Lasko, Damianos Karakos, Mahsa Yarmohammadi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2511.02246)  

**Abstract**: Recent research has shown that hallucinations, omissions, and biases are prevalent in everyday use-cases of LLMs. However, chatbots used in medical contexts must provide consistent advice in situations where non-medical factors are involved, such as when demographic information is present. In order to understand the conditions under which medical chatbots fail to perform as expected, we develop an infrastructure that 1) automatically generates queries to probe LLMs and 2) evaluates answers to these queries using multiple LLM-as-a-judge setups and prompts. For 1), our prompt creation pipeline samples the space of patient demographics, histories, disorders, and writing styles to create realistic questions that we subsequently use to prompt LLMs. In 2), our evaluation pipeline provides hallucination and omission detection using LLM-as-a-judge as well as agentic workflows, in addition to LLM-as-a-judge treatment category detectors. As a baseline study, we perform two case studies on inter-LLM agreement and the impact of varying the answering and evaluation LLMs. We find that LLM annotators exhibit low agreement scores (average Cohen's Kappa $\kappa=0.118$), and only specific (answering, evaluation) LLM pairs yield statistically significant differences across writing styles, genders, and races. We recommend that studies using LLM evaluation use multiple LLMs as evaluators in order to avoid arriving at statistically significant but non-generalizable results, particularly in the absence of ground-truth data. We also suggest publishing inter-LLM agreement metrics for transparency. Our code and dataset are available here: this https URL. 

**Abstract (ZH)**: 近期研究显示，LLM在日常使用场景中普遍存在幻觉、遗漏和偏差。然而，在医疗情境下的聊天机器人必须在涉及非医疗因素的情况下，如个人 demographics 信息存在时，提供一致的建议。为了理解医疗聊天机器人何时不能按预期执行，我们开发了一个基础设施，该基础设施包括1）自动生成查询以探查LLM，以及2）利用多个LLM作为评判者设置和提示进行答案评估。在1）中，我们的提示生成管道抽样患者 demographics、病史、疾病和写作风格的空间，以创建现实的问题，随后使用这些问题来提示LLM。在2）中，我们的评估管道利用LLM作为评判者进行幻觉和遗漏检出，并使用代理工作流程和LLM作为评判者的行为类别检测器进行评估。作为基准研究，我们在不同LLM之间进行两种案例研究，探讨了LLM之间的一致性以及回答和评估LLM的选择的影响。我们发现，LLM注释员的一致性评分为较低（平均科恩κ系数=0.118），仅特定（回答，评估）LLM配对在不同写作风格、性别和种族方面显示出统计学上的显著差异。我们建议在使用LLM评估的研究中使用多种LLM作为评判者，以避免得出统计学上有显著性但不具普适性的结果，特别是在缺乏真实数据的情况下。我们还建议公布LLM之间的一致性指标以提高透明度。我们的代码和数据集可在此获取：this https URL。 

---
# Structural Plasticity as Active Inference: A Biologically-Inspired Architecture for Homeostatic Control 

**Title (ZH)**: 结构塑性作为主动推断：一种生物学启发的稳态控制架构 

**Authors**: Brennen A. Hill  

**Link**: [PDF](https://arxiv.org/pdf/2511.02241)  

**Abstract**: Traditional neural networks, while powerful, rely on biologically implausible learning mechanisms such as global backpropagation. This paper introduces the Structurally Adaptive Predictive Inference Network (SAPIN), a novel computational model inspired by the principles of active inference and the morphological plasticity observed in biological neural cultures. SAPIN operates on a 2D grid where processing units, or cells, learn by minimizing local prediction errors. The model features two primary, concurrent learning mechanisms: a local, Hebbian-like synaptic plasticity rule based on the temporal difference between a cell's actual activation and its learned expectation, and a structural plasticity mechanism where cells physically migrate across the grid to optimize their information-receptive fields. This dual approach allows the network to learn both how to process information (synaptic weights) and also where to position its computational resources (network topology). We validated the SAPIN model on the classic Cart Pole reinforcement learning benchmark. Our results demonstrate that the architecture can successfully solve the CartPole task, achieving robust performance. The network's intrinsic drive to minimize prediction error and maintain homeostasis was sufficient to discover a stable balancing policy. We also found that while continual learning led to instability, locking the network's parameters after achieving success resulted in a stable policy. When evaluated for 100 episodes post-locking (repeated over 100 successful agents), the locked networks maintained an average 82% success rate. 

**Abstract (ZH)**: 传统的神经网络尽管强大，但依赖于生物不可行的学习机制，如全局反向传播。本文引入了结构自适应预测推理网络（SAPIN），这是一种受主动推理原则和生物神经网络中形态可塑性启发的新型计算模型。SAPIN 在一个二维网格上运行，其中处理单元或细胞通过最小化局部预测误差来学习。该模型包含两种主要的并发学习机制：基于细胞实际激活与学习期待之间的时间差的局部类Hebbian突触可塑性规则，以及一种结构可塑性机制，其中细胞在网格上物理迁移以优化其信息接收场。这种双重方法使网络能够学习如何处理信息（突触权重）以及如何定位其计算资源（网络拓扑）。我们使用经典的Cart Pole强化学习基准验证了SAPIN模型。实验结果表明，该架构能够成功解决CartPole任务，表现出稳定的表现。网络固有的减少预测误差和维持稳态的动力足以发现稳定平衡策略。我们还发现，持续学习会导致不稳定，而锁定网络参数并在成功后保持稳定则能够在100个成功代理进行100个回合的评估中维持平均82%的成功率。 

---
# LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation 

**Title (ZH)**: LACY：一种基于视觉-语言模型的语言行动循环 cycle用于自我提高的机器人 manipulation 

**Authors**: Youngjin Hong, Houjian Yu, Mingen Li, Changhyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02239)  

**Abstract**: Learning generalizable policies for robotic manipulation increasingly relies on large-scale models that map language instructions to actions (L2A). However, this one-way paradigm often produces policies that execute tasks without deeper contextual understanding, limiting their ability to generalize or explain their behavior. We argue that the complementary skill of mapping actions back to language (A2L) is essential for developing more holistic grounding. An agent capable of both acting and explaining its actions can form richer internal representations and unlock new paradigms for self-supervised learning. We introduce LACY (Language-Action Cycle), a unified framework that learns such bidirectional mappings within a single vision-language model. LACY is jointly trained on three synergistic tasks: generating parameterized actions from language (L2A), explaining observed actions in language (A2L), and verifying semantic consistency between two language descriptions (L2C). This enables a self-improving cycle that autonomously generates and filters new training data through an active augmentation strategy targeting low-confidence cases, thereby improving the model without additional human labels. Experiments on pick-and-place tasks in both simulation and the real world show that LACY improves task success rates by 56.46% on average and yields more robust language-action grounding for robotic manipulation. Project page: this https URL 

**Abstract (ZH)**: 基于语言-动作循环的学习框架：提升机器人操作中的语言动作映射能力 

---
# Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live 

**Title (ZH)**: 连续体：具有KV缓存时间生存期的高效稳健多轮LLM代理调度 

**Authors**: Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Alvin Cheung, Joseph Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2511.02230)  

**Abstract**: Agentic LLM applications interleave LLM generation requests with tool calls. These tool calls break the continuity of the workflow by creating pauses between LLM requests, bringing many challenges for the serving system, especially under multi-turn scenarios. Each pause potentially causes KV cache eviction and extra waiting time before entering the continuous batch for the following LLM request. Since these pauses happen for each call, this problem becomes increasingly severe as turn number grow for agentic programs. Previous works either fail to incorporate information from the tool call, evicting KV cache that leads to repetitive prefill or loading, or ignore the continuity of a multi-turn program, creating waiting time between turns that increases per-request latency.
We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by combining tool-aware KV cache timeout with program-level scheduling. By predicting tool call durations in agentic workflows, Continuum selectively pins the KV cache in GPU memory with a time-to-live value based on total turn number. When combined with program-level first-come-first-serve, Continuum prevents scheduling bubbles, preserves multi-turn continuity, and optimizes for throughput for complex agentic workflows. By modeling the variability of tool call and agent program continuity, Continuum outperforms state-of-the-art baselines. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B models shows that Continuum significantly improves the average job completion times, and remains performant across different hardware setups and DRAM offloading schemes. Preview code is available at: this https URL 

**Abstract (ZH)**: Continuum：一种通过工具意识型KV缓存超时与程序级调度结合来优化多轮代理工作负载完成时间的服务体系结构 

---
# Collaborative Attention and Consistent-Guided Fusion of MRI and PET for Alzheimer's Disease Diagnosis 

**Title (ZH)**: 基于MRI和PET的协作注意力和一致引导融合在阿尔茨海默病诊断中的应用 

**Authors**: Delin Ma, Menghui Zhou, Jun Qi, Yun Yang, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02228)  

**Abstract**: Alzheimer's disease (AD) is the most prevalent form of dementia, and its early diagnosis is essential for slowing disease progression. Recent studies on multimodal neuroimaging fusion using MRI and PET have achieved promising results by integrating multi-scale complementary features. However, most existing approaches primarily emphasize cross-modal complementarity while overlooking the diagnostic importance of modality-specific features. In addition, the inherent distributional differences between modalities often lead to biased and noisy representations, degrading classification performance. To address these challenges, we propose a Collaborative Attention and Consistent-Guided Fusion framework for MRI and PET based AD diagnosis. The proposed model introduces a learnable parameter representation (LPR) block to compensate for missing modality information, followed by a shared encoder and modality-independent encoders to preserve both shared and specific representations. Furthermore, a consistency-guided mechanism is employed to explicitly align the latent distributions across modalities. Experimental results on the ADNI dataset demonstrate that our method achieves superior diagnostic performance compared with existing fusion strategies. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是最常见的痴呆形式，早期诊断对于减缓疾病进展至关重要。近年来，融合MRI和PET的多模态神经成像研究通过整合多尺度互补特征取得了令人鼓舞的结果。然而，现有的大多数方法主要强调跨模态互补性，而忽视了模态特定特征的诊断重要性。此外，模态之间的固有分布差异往往会导致有偏和噪声表示，损害分类性能。为了解决这些挑战，我们提出了一种基于MRI和PET的协作注意力和一致导向融合框架，用于AD诊断。所提出的模型引入了一个可学习参数表示（LPR）块，以补偿缺失的模态信息，随后是一个共享编码器和模态独立编码器，以保留共享和特定的表示。此外，采用了一种一致性导向机制，以明确对齐模态之间的潜在分布。在ADNI数据集上的实验结果表明，与现有的融合策略相比，我们的方法在诊断性能上表现出优越性。 

---
# Optimizing Multi-Lane Intersection Performance in Mixed Autonomy Environments 

**Title (ZH)**: 混合自主环境中多车道交叉口性能优化 

**Authors**: Manonmani Sekar, Nasim Nezamoddini  

**Link**: [PDF](https://arxiv.org/pdf/2511.02217)  

**Abstract**: One of the main challenges in managing traffic at multilane intersections is ensuring smooth coordination between human-driven vehicles (HDVs) and connected autonomous vehicles (CAVs). This paper presents a novel traffic signal control framework that combines Graph Attention Networks (GAT) with Soft Actor-Critic (SAC) reinforcement learning to address this challenge. GATs are used to model the dynamic graph- structured nature of traffic flow to capture spatial and temporal dependencies between lanes and signal phases. The proposed SAC is a robust off-policy reinforcement learning algorithm that enables adaptive signal control through entropy-optimized decision making. This design allows the system to coordinate the signal timing and vehicle movement simultaneously with objectives focused on minimizing travel time, enhancing performance, ensuring safety, and improving fairness between HDVs and CAVs. The model is evaluated using a SUMO-based simulation of a four-way intersection and incorporating different traffic densities and CAV penetration rates. The experimental results demonstrate the effectiveness of the GAT-SAC approach by achieving a 24.1% reduction in average delay and up to 29.2% fewer traffic violations compared to traditional methods. Additionally, the fairness ratio between HDVs and CAVs improved to 1.59, indicating more equitable treatment across vehicle types. These findings suggest that the GAT-SAC framework holds significant promise for real-world deployment in mixed-autonomy traffic systems. 

**Abstract (ZH)**: 一种将图注意网络与Soft Actor-Critic强化学习结合的新型交通信号控制框架：实现人类驾驶车辆与连接自主车辆的协同管理 

---
# Adaptive Cooperative Transmission Design for Ultra-Reliable Low-Latency Communications via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的超可靠低延迟通信自适应协作传输设计 

**Authors**: Hyemin Yu, Hong-Chuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02216)  

**Abstract**: Next-generation wireless communication systems must support ultra-reliable low-latency communication (URLLC) service for mission-critical applications. Meeting stringent URLLC requirements is challenging, especially for two-hop cooperative communication. In this paper, we develop an adaptive transmission design for a two-hop relaying communication system. Each hop transmission adaptively configures its transmission parameters separately, including numerology, mini-slot size, and modulation and coding scheme, for reliable packet transmission within a strict latency constraint. We formulate the hop-specific transceiver configuration as a Markov decision process (MDP) and propose a dual-agent reinforcement learning-based cooperative latency-aware transmission (DRL-CoLA) algorithm to learn latency-aware transmission policies in a distributed manner. Simulation results verify that the proposed algorithm achieves the near-optimal reliability while satisfying strict latency requirements. 

**Abstract (ZH)**: 下一代无线通信系统必须支持使命关键型应用的超可靠低延迟通信（URLLC）服务。针对严苛的URLLC要求，特别是两跳协作通信，本文开发了一种适用于两跳中继通信系统的自适应传输设计方案。每个跳传输单独自适应配置其传输参数，包括数值参数、迷你时隙大小和调制编码方案，以在严格的延迟约束下实现可靠的数据包传输。我们将跳特定的收发机配置建模为马尔可夫决策过程（MDP），并提出了一种基于双Agent强化学习的分布式时延感知传输算法（DRL-CoLA），以学习时延感知的传输策略。仿真结果验证了所提算法在满足严格延迟要求的同时接近最优可靠性。 

---
# Estimation of Segmental Longitudinal Strain in Transesophageal Echocardiography by Deep Learning 

**Title (ZH)**: 基于深度学习的经食管超声心动图中超段长轴应变的估计 

**Authors**: Anders Austlid Taskén, Thierry Judge, Erik Andreas Rye Berg, Jinyang Yu, Bjørnar Grenne, Frank Lindseth, Svend Aakhus, Pierre-Marc Jodoin, Nicolas Duchateau, Olivier Bernard, Gabriel Kiss  

**Link**: [PDF](https://arxiv.org/pdf/2511.02210)  

**Abstract**: Segmental longitudinal strain (SLS) of the left ventricle (LV) is an important prognostic indicator for evaluating regional LV dysfunction, in particular for diagnosing and managing myocardial ischemia. Current techniques for strain estimation require significant manual intervention and expertise, limiting their efficiency and making them too resource-intensive for monitoring purposes. This study introduces the first automated pipeline, autoStrain, for SLS estimation in transesophageal echocardiography (TEE) using deep learning (DL) methods for motion estimation. We present a comparative analysis of two DL approaches: TeeFlow, based on the RAFT optical flow model for dense frame-to-frame predictions, and TeeTracker, based on the CoTracker point trajectory model for sparse long-sequence predictions.
As ground truth motion data from real echocardiographic sequences are hardly accessible, we took advantage of a unique simulation pipeline (SIMUS) to generate a highly realistic synthetic TEE (synTEE) dataset of 80 patients with ground truth myocardial motion to train and evaluate both models. Our evaluation shows that TeeTracker outperforms TeeFlow in accuracy, achieving a mean distance error in motion estimation of 0.65 mm on a synTEE test dataset.
Clinical validation on 16 patients further demonstrated that SLS estimation with our autoStrain pipeline aligned with clinical references, achieving a mean difference (95\% limits of agreement) of 1.09% (-8.90% to 11.09%). Incorporation of simulated ischemia in the synTEE data improved the accuracy of the models in quantifying abnormal deformation. Our findings indicate that integrating AI-driven motion estimation with TEE can significantly enhance the precision and efficiency of cardiac function assessment in clinical settings. 

**Abstract (ZH)**: 基于深度学习的自动管道自动Strain在经食管超声心动图中对左室段纵向应变的估计 

---
# Object-Centric 3D Gaussian Splatting for Strawberry Plant Reconstruction and Phenotyping 

**Title (ZH)**: 基于对象的草莓植物3D高斯绘制重建与表型分析 

**Authors**: Jiajia Li, Keyi Zhu, Qianwen Zhang, Dong Chen, Qi Sun, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02207)  

**Abstract**: Strawberries are among the most economically significant fruits in the United States, generating over $2 billion in annual farm-gate sales and accounting for approximately 13% of the total fruit production value. Plant phenotyping plays a vital role in selecting superior cultivars by characterizing plant traits such as morphology, canopy structure, and growth dynamics. However, traditional plant phenotyping methods are time-consuming, labor-intensive, and often destructive. Recently, neural rendering techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have emerged as powerful frameworks for high-fidelity 3D reconstruction. By capturing a sequence of multi-view images or videos around a target plant, these methods enable non-destructive reconstruction of complex plant architectures. Despite their promise, most current applications of 3DGS in agricultural domains reconstruct the entire scene, including background elements, which introduces noise, increases computational costs, and complicates downstream trait analysis. To address this limitation, we propose a novel object-centric 3D reconstruction framework incorporating a preprocessing pipeline that leverages the Segment Anything Model v2 (SAM-2) and alpha channel background masking to achieve clean strawberry plant reconstructions. This approach produces more accurate geometric representations while substantially reducing computational time. With a background-free reconstruction, our algorithm can automatically estimate important plant traits, such as plant height and canopy width, using DBSCAN clustering and Principal Component Analysis (PCA). Experimental results show that our method outperforms conventional pipelines in both accuracy and efficiency, offering a scalable and non-destructive solution for strawberry plant phenotyping. 

**Abstract (ZH)**: 美国草莓是最具经济意义的水果之一，年农场门市销售额超过20亿美元，占总水果生产价值的约13%。植物表型在选择优良品种方面起着关键作用，通过表征形态、冠层结构和生长动态等植物特征。然而，传统的植物表型方法耗时、劳动密集且往往具有破坏性。最近，神经渲染技术，尤其是神经辐射场（NeRF）和3D高斯点积（3DGS），已成为高保真3D重建的强大框架。通过拍摄目标植物周围的多视角图像或视频序列，这些方法能够实现非破坏性的复杂植物结构重建。尽管前景可人，但当前大多数3DGS在农业领域的应用都会重建整个场景，包括背景元素，这引入了噪声、增加了计算成本，并增加了后续特征分析的复杂性。为解决这一限制，我们提出了一种新的以对象为中心的3D重建框架，结合了利用Segment Anything Model v2（SAM-2）和alpha通道背景遮罩的预处理管道，以实现清晰的草莓植物重建。该方法提供了更准确的几何表示，并显著减少了计算时间。通过无背景的重建，我们的算法可以自动使用DBSCAN聚类和主成分分析（PCA）估计重要植物特征，如植物高度和冠层宽度。实验结果显示，我们的方法在准确性与效率上均优于传统管道，提供了一种可扩展且非破坏性的草莓植物表型解决方案。 

---
# Open the Oyster: Empirical Evaluation and Improvement of Code Reasoning Confidence in LLMs 

**Title (ZH)**: 开启牡蛎：LLM的代码推理置信度的实证评估与改进 

**Authors**: Shufan Wang, Xing Hu, Junkai Chen, Zhiyuan Pan, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2511.02197)  

**Abstract**: With the widespread application of large language models (LLMs) in the field of code intelligence, increasing attention has been paid to the reliability and controllability of their outputs in code reasoning tasks. Confidence estimation serves as an effective and convenient approach for evaluating these aspects. This paper proposes a confidence analysis and enhancement framework for LLMs tailored to code reasoning tasks. We conduct a comprehensive empirical study on the confidence reliability of mainstream LLMs across different tasks, and further evaluate the effectiveness of techniques such as prompt strategy optimisation and mathematical calibration (e.g., Platt Scaling) in improving confidence reliability. Our results show that DeepSeek-Reasoner achieves the best performance across various tasks, outperforming other models by up to $0.680$, $0.636$, and $13.652$ in terms of ECE, Brier Score, and Performance Score, respectively. The hybrid strategy combining the reassess prompt strategy and Platt Scaling achieves improvements of up to $0.541$, $0.628$, and $15.084$ over the original performance in the aforementioned three metrics. These results indicate that models with reasoning capabilities demonstrate superior confidence reliability, and that the hybrid strategy is the most effective in enhancing the confidence reliability of various models. Meanwhile, we elucidate the impact of different task complexities, model scales, and strategies on confidence performance, and highlight that the confidence of current LLMs in complex reasoning tasks still has considerable room for improvement. This study not only provides a research foundation and technical reference for the application of confidence in LLM-assisted software engineering, but also points the way for future optimisation and engineering deployment of confidence mechanisms. 

**Abstract (ZH)**: 大型语言模型在代码智能领域的广泛应用引起了人们对代码推理任务中其输出可靠性和可控性的高度重视。置信度评估作为一项有效且便捷的方法被广泛采用。本文提出了一种针对代码推理任务的大语言模型置信度分析与增强框架。我们对主流大语言模型在不同任务中的置信度可靠性进行了全面的经验研究，并进一步评估了诸如提示策略优化和数学校准（如Platt Scaling）等技术在提升置信度可靠性方面的有效性。结果显示，DeepSeek-Reasoner在各项任务中均表现出最佳性能，分别在ECE、Brier Score和Performance Score方面优于其他模型0.680、0.636和13.652。结合重新评估提示策略和Platt Scaling的混合策略在上述三项指标中分别实现了0.541、0.628和15.084的性能提升。这些结果表明，具备推理能力的模型在置信度可靠性方面表现出更优性能，而混合策略在多种模型上是提升置信度可靠性最有效的方法。同时，我们探讨了不同任务复杂性、模型规模和策略对置信度性能的影响，并指出当前大语言模型在复杂推理任务中的置信度仍有很大的提升空间。本文不仅为大语言模型辅助软件工程中置信度的应用提供了研究基础和技术参考，还为未来置信度机制的优化和工程部署指明了方向。 

---
# BoolSkeleton: Boolean Network Skeletonization via Homogeneous Pattern Reduction 

**Title (ZH)**: Bool骨架化：基于同质模式减少的布尔网络骨架化 

**Authors**: Liwei Ni, Jiaxi Zhang, Shenggen Zheng, Junfeng Liu, Xingyu Meng, Biwei Xie, Xingquan Li, Huawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02196)  

**Abstract**: Boolean equivalence allows Boolean networks with identical functionality to exhibit diverse graph structures. This gives more room for exploration in logic optimization, while also posing a challenge for tasks involving consistency between Boolean networks. To tackle this challenge, we introduce BoolSkeleton, a novel Boolean network skeletonization method that improves the consistency and reliability of design-specific evaluations. BoolSkeleton comprises two key steps: preprocessing and reduction. In preprocessing, the Boolean network is transformed into a defined Boolean dependency graph, where nodes are assigned the functionality-related status. Next, the homogeneous and heterogeneous patterns are defined for the node-level pattern reduction step. Heterogeneous patterns are preserved to maintain critical functionality-related dependencies, while homogeneous patterns can be reduced. Parameter K of the pattern further constrains the fanin size of these patterns, enabling fine-tuned control over the granularity of graph reduction. To validate BoolSkeleton's effectiveness, we conducted four analysis/downstream tasks around the Boolean network: compression analysis, classification, critical path analysis, and timing prediction, demonstrating its robustness across diverse scenarios. Furthermore, it improves above 55% in the average accuracy compared to the original Boolean network for the timing prediction task. These experiments underscore the potential of BoolSkeleton to enhance design consistency in logic synthesis. 

**Abstract (ZH)**: 布尔等价允许具有相同功能的布尔网络表现出多样的图结构。这为逻辑优化中的探索提供了更多空间，但也为涉及布尔网络一致性任务带来了挑战。为应对这一挑战，我们引入了BoolSkeleton——一种新颖的布尔网络骨架化方法，旨在提高设计特定评估的一致性和可靠性。BoolSkeleton包括两个关键步骤：预处理和约简。在预处理阶段，布尔网络被转换为定义好的布尔依赖图，节点被赋予与功能相关的状态。接着，定义了节点级模式约简中的同质模式和异质模式。异质模式被保留以保持关键的功能依赖关系，而同质模式可以被约简。模式参数K进一步限制了这些模式的扇入大小，从而使得对图约简粒度的调控更为精细。为验证BoolSkeleton的有效性，我们在布尔网络周围进行了四种分析/下游任务：压缩分析、分类、关键路径分析和定时预测，展示了其在多种场景下的稳健性。此外，与原始布尔网络相比，在定时预测任务中，它能够提高平均准确率超过55%。这些实验强调了BoolSkeleton在逻辑综合中增强设计一致性方面的潜力。 

---
# MM-UNet: Morph Mamba U-shaped Convolutional Networks for Retinal Vessel Segmentation 

**Title (ZH)**: MM-UNet：形态morph曼巴U形卷积网络用于视网膜血管分割 

**Authors**: Jiawen Liu, Yuanbo Zeng, Jiaming Liang, Yizhen Yang, Yiheng Zhang, Enhui Cai, Xiaoqi Sheng, Hongmin Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.02193)  

**Abstract**: Accurate detection of retinal vessels plays a critical role in reflecting a wide range of health status indicators in the clinical diagnosis of ocular diseases. Recently, advances in deep learning have led to a surge in retinal vessel segmentation methods, which have significantly contributed to the quantitative analysis of vascular morphology. However, retinal vasculature differs significantly from conventional segmentation targets in that it consists of extremely thin and branching structures, whose global morphology varies greatly across images. These characteristics continue to pose challenges to segmentation precision and robustness. To address these issues, we propose MM-UNet, a novel architecture tailored for efficient retinal vessel segmentation. The model incorporates Morph Mamba Convolution layers, which replace pointwise convolutions to enhance branching topological perception through morph, state-aware feature sampling. Additionally, Reverse Selective State Guidance modules integrate reverse guidance theory with state-space modeling to improve geometric boundary awareness and decoding efficiency. Extensive experiments conducted on two public retinal vessel segmentation datasets demonstrate the superior performance of the proposed method in segmentation accuracy. Compared to the existing approaches, MM-UNet achieves F1-score gains of 1.64 $\%$ on DRIVE and 1.25 $\%$ on STARE, demonstrating its effectiveness and advancement. The project code is public via this https URL. 

**Abstract (ZH)**: 准确检测视网膜血管在眼科疾病的临床诊断中反映了广泛的健康状态指标，对于反映多种健康状态指标至关重要。近期，深度学习的进步推动了视网膜血管分割方法的显著发展，极大地促进了血管形态的定量分析。然而，视网膜血管与传统分割目标显著不同，由极其细薄且分支的结构组成，其全局形态在不同图像中差异极大。这些特性继续对分割精度和鲁棒性构成挑战。为了解决这些问题，我们提出MM-UNet，一种专门用于高效视网膜血管分割的新架构。该模型采用Morph Mamba卷积层，用形态感知特征采样取代点wise卷积，增强分支拓扑感知。此外，Reverse Selective State Guidance模块结合逆向引导理论和状态空间建模，提高了几何边界意识和解码效率。在两个公开的视网膜血管分割数据集上的广泛实验表明，所提出的方法在分割准确性上表现出优异性能。与现有方法相比，MM-UNet在DRIVE上的F1得分提高了1.64%，在STARE上提高了1.25%，展示了其有效性和先进性。项目代码可通过此链接公开获取。 

---
# Tackling Incomplete Data in Air Quality Prediction: A Bayesian Deep Learning Framework for Uncertainty Quantification 

**Title (ZH)**: airs 质量预测中不完整数据的处理：基于不确定性量化的一种贝叶斯深度学习框架 

**Authors**: Yuzhuang Pian, Taiyu Wang, Shiqi Zhang, Rui Xu, Yonghong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02175)  

**Abstract**: Accurate air quality forecasts are vital for public health alerts, exposure assessment, and emissions control. In practice, observational data are often missing in varying proportions and patterns due to collection and transmission issues. These incomplete spatiotemporal records impede reliable inference and risk assessment and can lead to overconfident extrapolation. To address these challenges, we propose an end to end framework, the channel gated learning unit based spatiotemporal bayesian neural field (CGLUBNF). It uses Fourier features with a graph attention encoder to capture multiscale spatial dependencies and seasonal temporal dynamics. A channel gated learning unit, equipped with learnable activations and gated residual connections, adaptively filters and amplifies informative features. Bayesian inference jointly optimizes predictive distributions and parameter uncertainty, producing point estimates and calibrated prediction intervals. We conduct a systematic evaluation on two real world datasets, covering four typical missing data patterns and comparing against five state of the art baselines. CGLUBNF achieves superior prediction accuracy and sharper confidence intervals. In addition, we further validate robustness across multiple prediction horizons and analysis the contribution of extraneous variables. This research lays a foundation for reliable deep learning based spatio-temporal forecasting with incomplete observations in emerging sensing paradigms, such as real world vehicle borne mobile monitoring. 

**Abstract (ZH)**: 基于通道门控学习单元的空间时序贝叶斯神经场的准确空气质量预报方法 

---
# ScenicProver: A Framework for Compositional Probabilistic Verification of Learning-Enabled Systems 

**Title (ZH)**: ScenicProver: 一种学习使能系统组成化概率验证框架 

**Authors**: Eric Vin, Kyle A. Miller, Inigo Incer, Sanjit A. Seshia, Daniel J. Fremont  

**Link**: [PDF](https://arxiv.org/pdf/2511.02164)  

**Abstract**: Full verification of learning-enabled cyber-physical systems (CPS) has long been intractable due to challenges including black-box components and complex real-world environments. Existing tools either provide formal guarantees for limited types of systems or test the system as a monolith, but no general framework exists for compositional analysis of learning-enabled CPS using varied verification techniques over complex real-world environments. This paper introduces ScenicProver, a verification framework that aims to fill this gap. Built upon the Scenic probabilistic programming language, the framework supports: (1) compositional system description with clear component interfaces, ranging from interpretable code to black boxes; (2) assume-guarantee contracts over those components using an extension of Linear Temporal Logic containing arbitrary Scenic expressions; (3) evidence generation through testing, formal proofs via Lean 4 integration, and importing external assumptions; (4) systematic combination of generated evidence using contract operators; and (5) automatic generation of assurance cases tracking the provenance of system-level guarantees. We demonstrate the framework's effectiveness through a case study on an autonomous vehicle's automatic emergency braking system with sensor fusion. By leveraging manufacturer guarantees for radar and laser sensors and focusing testing efforts on uncertain conditions, our approach enables stronger probabilistic guarantees than monolithic testing with the same computational budget. 

**Abstract (ZH)**: 学习增强的控制物理系统（CPS）的全面验证长期以来由于黑盒组件和复杂的现实环境而难以实现。现有工具要么只能为有限类型的系统提供形式保证，要么将系统作为一个整体进行测试，但没有通用框架可以使用多种验证技术对学习增强的CPS进行组合分析。本文介绍了ScenicProver验证框架，旨在填补这一空白。该框架基于Scenic概率编程语言，支持：（1）具有清晰组件接口的组合系统描述，从可解释的代码到黑盒；（2）使用扩展的线性时序逻辑合同，包含任意Scenic表达式；（3）通过测试生成证据，通过Lean 4集成进行形式证明，并导入外部假设；（4）使用合同操作符系统组合生成的证据；以及（5）自动生成跟踪系统级保证源头的保障案。我们通过一个自动驾驶汽车自动紧急制动系统的传感器融合案例研究，展示了该框架的有效性。通过利用制造商对雷达和激光传感器的保证，以及在不确定条件下集中测试努力，我们的方法在相同的计算预算下提供了比整体测试更强的概率保证。 

---
# Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models 

**Title (ZH)**: 使用3D生成AI和视觉语言模型的文本到机器人多组件物体装配 

**Authors**: Alexander Htet Kyaw, Richa Gupta, Dhruv Shah, Anoop Sinha, Kory Mathewson, Stefanie Pender, Sachin Chitta, Yotto Koga, Faez Ahmed, Lawrence Sass, Randall Davis  

**Link**: [PDF](https://arxiv.org/pdf/2511.02162)  

**Abstract**: Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on object functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics. 

**Abstract (ZH)**: 基于视觉语言模型的3D生成AI在自然语言驱动的多组件物体机器人装配中的应用 

---
# Near Optimal Convergence to Coarse Correlated Equilibrium in General-Sum Markov Games 

**Title (ZH)**: _NEAR OPTIMAL CONVERGENCE TO COARSE CORRELATED EQUILIBRIUM IN GENERAL-SUM MARKOV GAMES_ 

**Authors**: Asrin Efe Yorulmaz, Tamer Başar  

**Link**: [PDF](https://arxiv.org/pdf/2511.02157)  

**Abstract**: No-regret learning dynamics play a central role in game theory, enabling decentralized convergence to equilibrium for concepts such as Coarse Correlated Equilibrium (CCE) or Correlated Equilibrium (CE). In this work, we improve the convergence rate to CCE in general-sum Markov games, reducing it from the previously best-known rate of $\mathcal{O}(\log^5 T / T)$ to a sharper $\mathcal{O}(\log T / T)$. This matches the best known convergence rate for CE in terms of $T$, number of iterations, while also improving the dependence on the action set size from polynomial to polylogarithmic-yielding exponential gains in high-dimensional settings. Our approach builds on recent advances in adaptive step-size techniques for no-regret algorithms in normal-form games, and extends them to the Markovian setting via a stage-wise scheme that adjusts learning rates based on real-time feedback. We frame policy updates as an instance of Optimistic Follow-the-Regularized-Leader (OFTRL), customized for value-iteration-based learning. The resulting self-play algorithm achieves, to our knowledge, the fastest known convergence rate to CCE in Markov games. 

**Abstract (ZH)**: No- regrets 学习动力学在一般和值马尔可夫博弈中以更快的速度收敛到粗略相关均衡 

---
# Disentangling Causal Substructures for Interpretable and Generalizable Drug Synergy Prediction 

**Title (ZH)**: 解构因果子结构以实现可解释性和普适性的药物协同效应预测 

**Authors**: Yi Luo, Haochen Zhao, Xiao Liang, Yiwei Liu, Yuye Zhang, Xinyu Li, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02146)  

**Abstract**: Drug synergy prediction is a critical task in the development of effective combination therapies for complex diseases, including cancer. Although existing methods have shown promising results, they often operate as black-box predictors that rely predominantly on statistical correlations between drug characteristics and results. To address this limitation, we propose CausalDDS, a novel framework that disentangles drug molecules into causal and spurious substructures, utilizing the causal substructure representations for predicting drug synergy. By focusing on causal sub-structures, CausalDDS effectively mitigates the impact of redundant features introduced by spurious substructures, enhancing the accuracy and interpretability of the model. In addition, CausalDDS employs a conditional intervention mechanism, where interventions are conditioned on paired molecular structures, and introduces a novel optimization objective guided by the principles of sufficiency and independence. Extensive experiments demonstrate that our method outperforms baseline models, particularly in cold start and out-of-distribution settings. Besides, CausalDDS effectively identifies key substructures underlying drug synergy, providing clear insights into how drug combinations work at the molecular level. These results underscore the potential of CausalDDS as a practical tool for predicting drug synergy and facilitating drug discovery. 

**Abstract (ZH)**: 药物协同效应预测是开发针对复杂疾病（包括癌症）的有效联合治疗方案的关键任务。尽管现有方法显示出有希望的结果，它们通常作为黑盒预测器运行，主要依赖于药物特性与结果之间的统计相关性。为解决这一局限性，我们提出了CausalDDS，这是一种新颖的框架，将药物分子解藕为因果性和伪因果性亚结构，利用因果性亚结构的表示来预测药物协同效应。通过关注因果性亚结构，CausalDDS有效地减轻了由伪因果性亚结构引入的冗余特征的影响，提高了模型的准确性和可解释性。此外，CausalDDS采用了一种条件干预机制，其中干预是基于配对的分子结构进行条件化的，并引入了一个由充分性和独立性原则指导的新优化目标。广泛的实验证明，我们在冷启动和分布外设置中均优于基线模型。此外，CausalDDS有效地识别了药物协同效应的关键亚结构，提供了对药物组合在分子水平上如何工作的清晰见解。这些结果突显了CausalDDS作为预测药物协同效应和促进药物发现的实用工具的潜力。 

---
# Matrix Sensing with Kernel Optimal Loss: Robustness and Optimization Landscape 

**Title (ZH)**: 核最优损失下的矩阵感知：稳健性与优化景觀 

**Authors**: Xinyuan Song, Jiaye Teng, Ziye Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.02122)  

**Abstract**: In this paper we study how the choice of loss functions of non-convex optimization problems affects their robustness and optimization landscape, through the study of noisy matrix sensing. In traditional regression tasks, mean squared error (MSE) loss is a common choice, but it can be unreliable for non-Gaussian or heavy-tailed noise. To address this issue, we adopt a robust loss based on nonparametric regression, which uses a kernel-based estimate of the residual density and maximizes the estimated log-likelihood. This robust formulation coincides with the MSE loss under Gaussian errors but remains stable under more general settings. We further examine how this robust loss reshapes the optimization landscape by analyzing the upper-bound of restricted isometry property (RIP) constants for spurious local minima to disappear. Through theoretical and empirical analysis, we show that this new loss excels at handling large noise and remains robust across diverse noise distributions. This work offers initial insights into enhancing the robustness of machine learning tasks through simply changing the loss, guided by an intuitive and broadly applicable analytical framework. 

**Abstract (ZH)**: 本文通过研究噪声矩阵感知问题，探讨非凸优化问题中损失函数的选择对其鲁棒性及优化景观的影响。在传统的回归任务中，均方误差（MSE）损失是一种常见选择，但对于非高斯或重尾噪声却不够可靠。为解决这一问题，本文采用基于非参数回归的鲁棒损失函数，该损失函数利用核估计的残差密度来最大化估计对数似然。这种鲁棒形式在高斯误差情况下与MSE损失一致，但在更广泛的设置下仍能保持稳定性。进一步通过分析有限约束等距性质（RIP）常数的上界，研究这种鲁棒损失如何重塑优化景观，使无效局部极小值消失。通过理论和实证分析，本文显示这种新损失函数在处理大规模噪声和多种噪声分布时表现出色且鲁棒。本文提供了有关通过简单改变损失函数来增强机器学习任务鲁棒性的初步见解，基于一个直观且广泛适用的分析框架。 

---
# Metamorphic Testing of Large Language Models for Natural Language Processing 

**Title (ZH)**: 大型语言模型的 metamorphic 测试在自然语言处理中的应用 

**Authors**: Steven Cho, Stefano Ruberto, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2511.02108)  

**Abstract**: Using large language models (LLMs) to perform natural language processing (NLP) tasks has become increasingly pervasive in recent times. The versatile nature of LLMs makes them applicable to a wide range of such tasks. While the performance of recent LLMs is generally outstanding, several studies have shown that they can often produce incorrect results. Automatically identifying these faulty behaviors is extremely useful for improving the effectiveness of LLMs. One obstacle to this is the limited availability of labeled datasets, which necessitates an oracle to determine the correctness of LLM behaviors. Metamorphic testing (MT) is a popular testing approach that alleviates this oracle problem. At the core of MT are metamorphic relations (MRs), which define relationships between the outputs of related inputs. MT can expose faulty behaviors without the need for explicit oracles (e.g., labeled datasets). This paper presents the most comprehensive study of MT for LLMs to date. We conducted a literature review and collected 191 MRs for NLP tasks. We implemented a representative subset (36 MRs) to conduct a series of experiments with three popular LLMs, running approximately 560,000 metamorphic tests. The results shed light on the capabilities and opportunities of MT for LLMs, as well as its limitations. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）执行自然语言处理（NLP）任务在当今越来越普遍。LLMs的多功能性使它们适用于广泛的任务。尽管近年来LLMs的性能通常非常出色，但多项研究显示它们经常会产生错误结果。自动识别这些故障行为对于提高LLMs的有效性极为有用。这一挑战的一个障碍是标记数据集的有限可用性，这需要一个知识渊博的实体来确定LLM行为的正确性。元型测试（MT）是一种流行的方法，可以缓解这一问题。MT的核心是元型关系（MRs），它们定义了相关输入输出之间的关系。MT可以在不需要显式或acles（例如，标记数据集）的情况下揭示故障行为。本文进行了迄今为止对于LLMs最全面的MT研究。我们进行了文献综述，并收集了191个适用于NLP任务的MRs。我们实现了一个代表性子集（36个MRs）以对三种流行LLMs进行一系列实验，共运行了约560,000个元型测试。结果揭示了MT在LLMs中的能力和潜力，以及其局限性。 

---
# Geometric Data Valuation via Leverage Scores 

**Title (ZH)**: 几何数据估值通过杠杆得分 

**Authors**: Rodrigo Mendoza-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.02100)  

**Abstract**: Shapley data valuation provides a principled, axiomatic framework for assigning importance to individual datapoints, and has gained traction in dataset curation, pruning, and pricing. However, it is a combinatorial measure that requires evaluating marginal utility across all subsets of the data, making it computationally infeasible at scale. We propose a geometric alternative based on statistical leverage scores, which quantify each datapoint's structural influence in the representation space by measuring how much it extends the span of the dataset and contributes to the effective dimensionality of the training problem. We show that our scores satisfy the dummy, efficiency, and symmetry axioms of Shapley valuation and that extending them to \emph{ridge leverage scores} yields strictly positive marginal gains that connect naturally to classical A- and D-optimal design criteria. We further show that training on a leverage-sampled subset produces a model whose parameters and predictive risk are within $O(\varepsilon)$ of the full-data optimum, thereby providing a rigorous link between data valuation and downstream decision quality. Finally, we conduct an active learning experiment in which we empirically demonstrate that ridge-leverage sampling outperforms standard baselines without requiring access gradients or backward passes. 

**Abstract (ZH)**: 基于统计杠杆得分的几何替代方案：与舍普利数据估值的关联及在主动学习中的应用 

---
# Uncertainty Guided Online Ensemble for Non-stationary Data Streams in Fusion Science 

**Title (ZH)**: 面向融合科学中非平稳数据流的不确定性引导在线集成方法 

**Authors**: Kishansingh Rajput, Malachi Schram, Brian Sammuli, Sen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.02092)  

**Abstract**: Machine Learning (ML) is poised to play a pivotal role in the development and operation of next-generation fusion devices. Fusion data shows non-stationary behavior with distribution drifts, resulted by both experimental evolution and machine wear-and-tear. ML models assume stationary distribution and fail to maintain performance when encountered with such non-stationary data streams. Online learning techniques have been leveraged in other domains, however it has been largely unexplored for fusion applications. In this paper, we present an application of online learning to continuously adapt to drifting data stream for prediction of Toroidal Field (TF) coils deflection at the DIII-D fusion facility. The results demonstrate that online learning is critical to maintain ML model performance and reduces error by 80% compared to a static model. Moreover, traditional online learning can suffer from short-term performance degradation as ground truth is not available before making the predictions. As such, we propose an uncertainty guided online ensemble method to further improve the performance. The Deep Gaussian Process Approximation (DGPA) technique is leveraged for calibrated uncertainty estimation and the uncertainty values are then used to guide a meta-algorithm that produces predictions based on an ensemble of learners trained on different horizon of historical data. The DGPA also provides uncertainty estimation along with the predictions for decision makers. The online ensemble and the proposed uncertainty guided online ensemble reduces predictions error by about 6%, and 10% respectively over standard single model based online learning. 

**Abstract (ZH)**: 机器学习在下一代聚变装置开发与运行中的关键作用：基于在线学习方法的托量磁场线圈偏移预测 

---
# Natural Building Blocks for Structured World Models: Theory, Evidence, and Scaling 

**Title (ZH)**: 结构化世界模型的自然构建块：理论、证据与扩展 

**Authors**: Lancelot Da Costa, Sanjeev Namjoshi, Mohammed Abbas Ansari, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2511.02091)  

**Abstract**: The field of world modeling is fragmented, with researchers developing bespoke architectures that rarely build upon each other. We propose a framework that specifies the natural building blocks for structured world models based on the fundamental stochastic processes that any world model must capture: discrete processes (logic, symbols) and continuous processes (physics, dynamics); the world model is then defined by the hierarchical composition of these building blocks. We examine Hidden Markov Models (HMMs) and switching linear dynamical systems (sLDS) as natural building blocks for discrete and continuous modeling--which become partially-observable Markov decision processes (POMDPs) and controlled sLDS when augmented with actions. This modular approach supports both passive modeling (generation, forecasting) and active control (planning, decision-making) within the same architecture. We avoid the combinatorial explosion of traditional structure learning by largely fixing the causal architecture and searching over only four depth parameters. We review practical expressiveness through multimodal generative modeling (passive) and planning from pixels (active), with performance competitive to neural approaches while maintaining interpretability. The core outstanding challenge is scalable joint structure-parameter learning; current methods finesse this by cleverly growing structure and parameters incrementally, but are limited in their scalability. If solved, these natural building blocks could provide foundational infrastructure for world modeling, analogous to how standardized layers enabled progress in deep learning. 

**Abstract (ZH)**: 世界建模领域支离破碎，研究人员开发的专用架构很少互相借鉴。我们提出了一种框架，基于任何世界模型必须捕获的基本随机过程——离散过程（逻辑、符号）和连续过程（物理、动力学）——来规定结构化世界模型的自然构建块。然后，世界模型通过这些构建块的分层组合来定义。我们研究隐藏马尔可夫模型（HMMs）和切换线性动态系统（sLDS）作为离散和连续建模的自然构建块——当增加动作时，它们成为部分可观测马尔可夫决策过程（POMDPs）和受控sLDS。这种模块化方法同时支持被动建模（生成、预测）和主动控制（规划、决策）。我们通过多模态生成建模（被动）和从像素进行规划（主动）来避免传统结构学习中的组合爆炸，通过主要固定因果结构并在仅四个深度参数上进行搜索来实现。我们通过实用的表达能力验证了这一点，表现可与神经方法相当且保持可解释性。核心挑战是可扩展的联合结构参数学习；现有方法通过巧妙地逐步增长结构和参数来处理这一挑战，但在可扩展性方面受到限制。如果得到解决，这些自然构建块可以为世界建模提供基础结构，类似于标准化层如何推动深度学习的进步。 

---
# Energy Loss Functions for Physical Systems 

**Title (ZH)**: 物理系统中的能量损失函数 

**Authors**: Sékou-Oumar Kaba, Kusha Sareen, Daniel Levy, Siamak Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2511.02087)  

**Abstract**: Effectively leveraging prior knowledge of a system's physics is crucial for applications of machine learning to scientific domains. Previous approaches mostly focused on incorporating physical insights at the architectural level. In this paper, we propose a framework to leverage physical information directly into the loss function for prediction and generative modeling tasks on systems like molecules and spins. We derive energy loss functions assuming that each data sample is in thermal equilibrium with respect to an approximate energy landscape. By using the reverse KL divergence with a Boltzmann distribution around the data, we obtain the loss as an energy difference between the data and the model predictions. This perspective also recasts traditional objectives like MSE as energy-based, but with a physically meaningless energy. In contrast, our formulation yields physically grounded loss functions with gradients that better align with valid configurations, while being architecture-agnostic and computationally efficient. The energy loss functions also inherently respect physical symmetries. We demonstrate our approach on molecular generation and spin ground-state prediction and report significant improvements over baselines. 

**Abstract (ZH)**: 有效利用系统的物理先验知识对于将机器学习应用于科学领域至关重要。以往的方法主要侧重于在架构层面融入物理洞见。本文提出了一种框架，直接将物理信息纳入损失函数中，用于分子和自旋等系统上的预测和生成建模任务。我们假设每个数据样本相对于近似能量景观处于热平衡状态，推导出能量损失函数。通过使用数据周围的玻尔兹曼分布的反向KL散度，我们获得损失为数据与模型预测之间的能量差。这一视角还将传统目标如均方误差重新解释为能量基的，但能量在物理上并无意义。相比之下，我们的形式化导出了基于物理的损失函数，其梯度更好地与有效配置对齐，同时具有架构无关性和计算效率。能量损失函数还固有地尊重物理对称性。我们分别在分子生成和自旋基态预测上展示了该方法，并报告了相对于基线的显著改进。 

---
# Watermarking Discrete Diffusion Language Models 

**Title (ZH)**: 离散扩散语言模型中的水印技术 

**Authors**: Avi Bagchi, Akhil Bhimaraju, Moulik Choraria, Daniel Alabi, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2511.02083)  

**Abstract**: Watermarking has emerged as a promising technique to track AI-generated content and differentiate it from authentic human creations. While prior work extensively studies watermarking for autoregressive large language models (LLMs) and image diffusion models, none address discrete diffusion language models, which are becoming popular due to their high inference throughput. In this paper, we introduce the first watermarking method for discrete diffusion models by applying the distribution-preserving Gumbel-max trick at every diffusion step and seeding the randomness with the sequence index to enable reliable detection. We experimentally demonstrate that our scheme is reliably detectable on state-of-the-art diffusion language models and analytically prove that it is distortion-free with an exponentially decaying probability of false detection in the token sequence length. 

**Abstract (ZH)**: 水印技术作为一种追踪AI生成内容并与真实人类创作区分的有前途的方法已经 emergence 。尽管先前的工作对自回归大型语言模型和图像扩散模型的水印技术进行了广泛研究，但尚未有工作针对成为流行趋势的高推理吞吐量离散扩散语言模型。在本文中，我们通过在每次扩散步骤中应用分布保持的Gumbel-max技巧并将随机性与序列索引结合，首次提出了离散扩散模型的水印方法，以实现可靠的检测。我们实验性地证明了该方案在最先进的扩散语言模型上具有可靠的可检测性，并从理论上证明在令牌序列长度上，该方案具有指数衰减的误检概率且无失真。 

---
# Vortex: Hosting ML Inference and Knowledge Retrieval Services With Tight Latency and Throughput Requirements 

**Title (ZH)**: 涡旋：满足紧密延迟和吞吐量要求的机器学习推理和知识检索服务-hosting 

**Authors**: Yuting Yang, Tiancheng Yuan, Jamal Hashim, Thiago Garrett, Jeffrey Qian, Ann Zhang, Yifan Wang, Weijia Song, Ken Birman  

**Link**: [PDF](https://arxiv.org/pdf/2511.02062)  

**Abstract**: There is growing interest in deploying ML inference and knowledge retrieval as services that could support both interactive queries by end users and more demanding request flows that arise from AIs integrated into a end-user applications and deployed as agents. Our central premise is that these latter cases will bring service level latency objectives (SLOs). Existing ML serving platforms use batching to optimize for high throughput, exposing them to unpredictable tail latencies. Vortex enables an SLO-first approach. For identical tasks, Vortex's pipelines achieve significantly lower and more stable latencies than TorchServe and Ray Serve over a wide range of workloads, often enabling a given SLO target at more than twice the request rate. When RDMA is available, the Vortex advantage is even more significant. 

**Abstract (ZH)**: 随着对将机器学习推理和知识检索部署为服务的兴趣不断增强，这些服务既能支持最终用户的手动查询，也能支持集成到最终用户应用程序中的AI所产生更具挑战性的请求流。我们的核心假设是，后者将带来服务级别延迟目标（SLO）。现有的机器学习服务平台通过批量处理来优化高吞吐量，从而暴露在不可预测的尾部延迟风险中。Vortex 使SLO优先的方法成为可能。对于相同的任务，Vortex的管道在各种工作负载范围内实现显著更低且更稳定的延迟，经常能够在两倍于请求率的情况下达到给定的SLO目标。当可用时，基于RDMA的优势更加显著。 

---
# Text-VQA Aug: Pipelined Harnessing of Large Multimodal Models for Automated Synthesis 

**Title (ZH)**: Text-VQA 增强：流水线集成大规模多模态模型进行自动化合成 

**Authors**: Soham Joshi, Shwet Kamal Mishra, Viswanath Gopalakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02046)  

**Abstract**: Creation of large-scale databases for Visual Question Answering tasks pertaining to the text data in a scene (text-VQA) involves skilful human annotation, which is tedious and challenging. With the advent of foundation models that handle vision and language modalities, and with the maturity of OCR systems, it is the need of the hour to establish an end-to-end pipeline that can synthesize Question-Answer (QA) pairs based on scene-text from a given image. We propose a pipeline for automated synthesis for text-VQA dataset that can produce faithful QA pairs, and which scales up with the availability of scene text data. Our proposed method harnesses the capabilities of multiple models and algorithms involving OCR detection and recognition (text spotting), region of interest (ROI) detection, caption generation, and question generation. These components are streamlined into a cohesive pipeline to automate the synthesis and validation of QA pairs. To the best of our knowledge, this is the first pipeline proposed to automatically synthesize and validate a large-scale text-VQA dataset comprising around 72K QA pairs based on around 44K images. 

**Abstract (ZH)**: 基于场景文本的视觉问答任务（text-VQA）的大规模数据库创建涉及细致的人工标注，这既繁琐又具有挑战性。随着能够处理视觉和语言模态的基础模型的出现，以及光学字符识别（OCR）系统的发展成熟，建立一个能够基于给定图像的场景文本自动生成问题-答案（QA）对的端到端管道迫在眉睫。我们提出了一种自动合成pipeline，用于生成忠实的QA对，并且随着场景文本数据的增加而扩展。我们提出的方法利用了OCR检测和识别（文本检测）、兴趣区域（ROI）检测、Caption生成和问题生成等多种模型和算法的能力。这些组件被整合成一个协调的pipeline，以自动化QA对的合成和验证。据我们所知，这是首次提出一个自动合成和验证基于约44K图像生成约72K QA对的大规模text-VQA数据集的pipeline。 

---
# Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning 

**Title (ZH)**: 通过推理正则化：通过解释增强微调在语言模型分类中的系统性改进 

**Authors**: Vivswan Shah, Randy Cogill, Hanwei Yue, Gopinath Chennupati, Rinat Khaziev  

**Link**: [PDF](https://arxiv.org/pdf/2511.02044)  

**Abstract**: Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines.
A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts.
Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference. 

**Abstract (ZH)**: 细调LLMs进行分类通常将输入直接映射到标签。我们询问在细调过程中为每个标签附上简短解释是否能获得更好的模型。我们从对话响应质量的三个维度进行评估：自然度、完备性和主题相关性，每个维度按5点量表评分。使用多个LLM生成的集成数据，我们细调了一个7亿参数的模型，并在六个多样化的对话数据集中进行测试。在18个数据集和任务设置中，带有解释的标签训练优于仅标签的基础模型。

一个中心且意外的结果涉及随机标记。我们用与原有人编写解释在词汇上对齐但语义上不连贯的文本（例如，洗牌或词袋变体）替换人工撰写的解释。尽管缺乏语义，这些伪解释仍然在仅标签训练的基础上提高了精度，并且常常缩小了与真实解释之间差距的大部分。这种效应在不同数据集和训练种子中持续存在，表明收益主要来自结构而非意义：额外的标记预算促进了更丰富的中间计算，并作为一种正则化手段减少了过于自信的捷径。

内部分析支持这一观点：增强解释的模型在中间层表现出更高的激活熵，并且在输出层具有更锐利的预测质量，这与在决策前增加的斟酌一致。总体而言，无论是真实理由还是精心构建的随机标记序列的增强解释，都可以改善LLM分类的准确性和可靠性，并阐明标记级别架构如何在推理过程中塑造计算。 

---
# Quantum-Enhanced Generative Models for Rare Event Prediction 

**Title (ZH)**: 量子增强生成模型在稀有事件预测中的应用 

**Authors**: M.Z. Haider, M.U. Ghouri, Tayyaba Noreen, M. Salman  

**Link**: [PDF](https://arxiv.org/pdf/2511.02042)  

**Abstract**: Rare events such as financial crashes, climate extremes, and biological anomalies are notoriously difficult to model due to their scarcity and heavy-tailed distributions. Classical deep generative models often struggle to capture these rare occurrences, either collapsing low-probability modes or producing poorly calibrated uncertainty estimates. In this work, we propose the Quantum-Enhanced Generative Model (QEGM), a hybrid classical-quantum framework that integrates deep latent-variable models with variational quantum circuits. The framework introduces two key innovations: (1) a hybrid loss function that jointly optimizes reconstruction fidelity and tail-aware likelihood, and (2) quantum randomness-driven noise injection to enhance sample diversity and mitigate mode collapse. Training proceeds via a hybrid loop where classical parameters are updated through backpropagation while quantum parameters are optimized using parameter-shift gradients. We evaluate QEGM on synthetic Gaussian mixtures and real-world datasets spanning finance, climate, and protein structure. Results demonstrate that QEGM reduces tail KL divergence by up to 50 percent compared to state-of-the-art baselines (GAN, VAE, Diffusion), while improving rare-event recall and coverage calibration. These findings highlight the potential of QEGM as a principled approach for rare-event prediction, offering robustness beyond what is achievable with purely classical methods. 

**Abstract (ZH)**: 量子增强生成模型：针对罕见事件的混合经典-量子框架 

---
# RobustFSM: Submodular Maximization in Federated Setting with Malicious Clients 

**Title (ZH)**: RobustFSM: 在存在恶意客户端的联邦设置下进行子模态最大化 

**Authors**: Duc A. Tran, Dung Truong, Duy Le  

**Link**: [PDF](https://arxiv.org/pdf/2511.02029)  

**Abstract**: Submodular maximization is an optimization problem benefiting many machine learning applications, where we seek a small subset best representing an extremely large dataset. We focus on the federated setting where the data are locally owned by decentralized clients who have their own definitions for the quality of representability. This setting requires repetitive aggregation of local information computed by the clients. While the main motivation is to respect the privacy and autonomy of the clients, the federated setting is vulnerable to client misbehaviors: malicious clients might share fake information. An analogy is backdoor attack in conventional federated learning, but our challenge differs freshly due to the unique characteristics of submodular maximization. We propose RobustFSM, a federated submodular maximization solution that is robust to various practical client attacks. Its performance is substantiated with an empirical evaluation study using real-world datasets. Numerical results show that the solution quality of RobustFSM substantially exceeds that of the conventional federated algorithm when attacks are severe. The degree of this improvement depends on the dataset and attack scenarios, which can be as high as 200% 

**Abstract (ZH)**: 子模 Clips 极大化是一种优化问题，广泛应用于许多机器学习应用中，目标是从极大数据集中找到一个小子集最好地代表整个数据集。本文关注数据由分散客户端本地拥有的联邦设置，在这种设置下，每个客户端都有自己的代表质量定义。该设置要求重复聚合客户端计算的本地信息。主要动机是尊重客户端的隐私和自主性，但联邦设置也容易遭受客户端的不良行为：恶意客户端可能分享虚假信息。这一挑战由于子模 Clips 极大化的独特特性而区别于传统的联邦学习后门攻击。我们提出了 RobustFSM，一种对各种实际客户端攻击具有鲁棒性的联邦子模 Clips 极大化解决方案，并通过使用真实数据集的实证评估研究验证了其性能。数值结果显示，当攻击严重时，RobustFSM 的解的质量明显优于传统的联邦算法，这种改进的程度取决于数据集和攻击场景，最高可达 200%。 

---
# Path-Coordinated Continual Learning with Neural Tangent Kernel-Justified Plasticity: A Theoretical Framework with Near State-of-the-Art Performance 

**Title (ZH)**: 基于神经切线核验证的路径协调连续学习与神经可塑性：一种接近最先进性能的理论框架 

**Authors**: Rathin Chandra Shit  

**Link**: [PDF](https://arxiv.org/pdf/2511.02025)  

**Abstract**: Catastrophic forgetting is one of the fundamental issues of continual learning because neural networks forget the tasks learned previously when trained on new tasks. The proposed framework is a new path-coordinated framework of continual learning that unites the Neural Tangent Kernel (NTK) theory of principled plasticity bounds, statistical validation by Wilson confidence intervals, and evaluation of path quality by the use of multiple metrics. Experimental evaluation shows an average accuracy of 66.7% at the cost of 23.4% catastrophic forgetting on Split-CIFAR10, a huge improvement over the baseline and competitive performance achieved, which is very close to state-of-the-art results. Further, it is found out that NTK condition numbers are predictive indicators of learning capacity limits, showing the existence of a critical threshold at condition number $>10^{11}$. It is interesting to note that the proposed strategy shows a tendency of lowering forgetting as the sequence of tasks progresses (27% to 18%), which is a system stabilization. The framework validates 80% of discovered paths with a rigorous statistical guarantee and maintains 90-97% retention on intermediate tasks. The core capacity limits of the continual learning environment are determined in the analysis, and actionable insights to enhance the adaptive regularization are offered. 

**Abstract (ZH)**: 连续学习中的灾难性遗忘是一个基本问题，因为神经网络在学习新任务时会忘记之前学习的任务。本文提出的框架是一个新的路径协调连续学习框架，结合了原理性可塑性界限的神经切线核（NTK）理论、威尔逊置信区间统计验证以及通过多种指标评估路径质量。实验评估显示，在Split-CIFAR10数据集上平均准确率为66.7%，灾难性遗忘的成本为23.4%，相对于基线取得了巨大的提升，并且达到了可竞争的性能水平，非常接近当前最先进的结果。进一步发现，NTK条件数是学习容量限制的预测指标，显示出在条件数大于$10^{11}$时存在一个关键阈值。值得注意的是，所提出的策略在任务序列进展中显示出降低遗忘的趋势（从27%到18%），这是一种系统稳定化现象。该框架以严格的统计保证验证了80%的发现路径，并在中间任务上保持了90-97%的记忆保留率。分析确定了连续学习环境的核心容量限制，并提供了增强自适应正则化的可操作见解。 

---
# Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior 

**Title (ZH)**: Emergently Misaligned行为中的共享参数子空间与跨任务线性关系 

**Authors**: Daniel Aarao Reis Arturi, Eric Zhang, Andrew Ansah, Kevin Zhu, Ashwinee Panda, Aishwarya Balwani  

**Link**: [PDF](https://arxiv.org/pdf/2511.02022)  

**Abstract**: Recent work has discovered that large language models can develop broadly misaligned behaviors after being fine-tuned on narrowly harmful datasets, a phenomenon known as emergent misalignment (EM). However, the fundamental mechanisms enabling such harmful generalization across disparate domains remain poorly understood. In this work, we adopt a geometric perspective to study EM and demonstrate that it exhibits a fundamental cross-task linear structure in how harmful behavior is encoded across different datasets. Specifically, we find a strong convergence in EM parameters across tasks, with the fine-tuned weight updates showing relatively high cosine similarities, as well as shared lower-dimensional subspaces as measured by their principal angles and projection overlaps. Furthermore, we also show functional equivalence via linear mode connectivity, wherein interpolated models across narrow misalignment tasks maintain coherent, broadly misaligned behavior. Our results indicate that EM arises from different narrow tasks discovering the same set of shared parameter directions, suggesting that harmful behaviors may be organized into specific, predictable regions of the weight landscape. By revealing this fundamental connection between parametric geometry and behavioral outcomes, we hope our work catalyzes further research on parameter space interpretability and weight-based interventions. 

**Abstract (ZH)**: 近年来的研究发现，大型语言模型在狭义有害数据集上微调后，可能会表现出广泛 misaligned 的行为，这一现象被称为 emergent misalignment (EM)。然而，促使此类有害泛化的根本机制在不同领域之间仍不清楚。在本项工作中，我们采用几何视角研究 EM，并证明它在不同数据集上如何编码有害行为方面表现出一种基本的跨任务线性结构。具体来说，我们发现不同任务的 EM 参数存在强烈的收敛性，微调权重更新显示出较高的余弦相似度，并且存在共享的低维子空间。此外，我们还通过线性模式连通性展示了功能等价性，在狭窄 misalignment 任务之间的插值模型保持一致且广泛 misaligned 的行为。我们的结果表明，EM 是不同狭义任务找到相同参数方向集的结果，暗示有害行为可能被组织在权重景观的特定、可预测区域中。通过揭示参数几何学与行为结果之间的基本联系，我们希望本项工作能促进参数空间可解释性和基于权重的干预措施的研究。 

---
# InteracSPARQL: An Interactive System for SPARQL Query Refinement Using Natural Language Explanations 

**Title (ZH)**: InteracSPARQL：基于自然语言解释的SPARQL查询改进交互系统 

**Authors**: Xiangru Jian, Zhengyuan Dong, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02002)  

**Abstract**: In recent years, querying semantic web data using SPARQL has remained challenging, especially for non-expert users, due to the language's complex syntax and the prerequisite of understanding intricate data structures. To address these challenges, we propose InteracSPARQL, an interactive SPARQL query generation and refinement system that leverages natural language explanations (NLEs) to enhance user comprehension and facilitate iterative query refinement. InteracSPARQL integrates LLMs with a rule-based approach to first produce structured explanations directly from SPARQL abstract syntax trees (ASTs), followed by LLM-based linguistic refinements. Users can interactively refine queries through direct feedback or LLM-driven self-refinement, enabling the correction of ambiguous or incorrect query components in real time. We evaluate InteracSPARQL on standard benchmarks, demonstrating significant improvements in query accuracy, explanation clarity, and overall user satisfaction compared to baseline approaches. Our experiments further highlight the effectiveness of combining rule-based methods with LLM-driven refinements to create more accessible and robust SPARQL interfaces. 

**Abstract (ZH)**: 近年来，使用SPARQL查询语义网数据仍具有挑战性，尤其是对于非专家用户而言，由于该语言复杂的语法结构和对复杂数据结构的理解要求。为解决这些挑战，我们提出了一种名为InteracSPARQL的交互式SPARQL查询生成和细化系统，该系统利用自然语言解释（NLE）来增强用户理解并促进迭代查询细化。InteracSPARQL结合了基于规则的方法和大语言模型（LLMs），首先从SPARQL抽象语法树（ASTs）直接生成结构化的解释，然后进行基于LLM的语言细化。用户可以通过直接反馈或LLM驱动的自我细化来交互式地细化查询，从而实时纠正模糊或错误的查询成分。我们在标准基准上评估了InteracSPARQL，与基础方法相比，结果显示在查询准确性、解释清晰度和整体用户满意度方面取得了显著改进。我们的实验进一步强调了结合基于规则的方法和LLM驱动的细化在创建更易于访问和稳健的SPARQL界面方面的有效性。 

---
# TRACE: Textual Reasoning for Affordance Coordinate Extraction 

**Title (ZH)**: 轨迹：基于文本的推理以提取功能坐标 

**Authors**: Sangyun Park, Jin Kim, Yuchen Cui, Matthew S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2511.01999)  

**Abstract**: Vision-Language Models (VLMs) struggle to translate high-level instructions into the precise spatial affordances required for robotic manipulation. While visual Chain-of-Thought (CoT) methods exist, they are often computationally intensive. In this work, we introduce TRACE (Textual Reasoning for Affordance Coordinate Extraction), a novel methodology that integrates a textual Chain of Reasoning (CoR) into the affordance prediction process. We use this methodology to create the TRACE dataset, a large-scale collection created via an autonomous pipeline that pairs instructions with explicit textual rationales. By fine-tuning a VLM on this data, our model learns to externalize its spatial reasoning before acting. Our experiments show that our TRACE-tuned model achieves state-of-the-art performance, reaching 48.1% accuracy on the primary Where2Place (W2P) benchmark (a 9.6% relative improvement) and 55.0% on the more challenging W2P(h) subset. Crucially, an ablation study demonstrates that performance scales directly with the amount of reasoning data used, confirming the CoR's effectiveness. Furthermore, analysis of the model's attention maps reveals an interpretable reasoning process where focus shifts dynamically across reasoning steps. This work shows that training VLMs to generate a textual CoR is an effective and robust strategy for enhancing the precision, reliability, and interpretability of VLM-based robot control. Our dataset and code are available at this https URL 

**Abstract (ZH)**: Vision-Language模型在将高层次指令转化为精确的空间操作能力方面存在挑战。虽然视觉链式思考方法已存在，但它们通常计算成本高。在本工作中，我们引入了TRACE（文本推理以提取操作坐标），这是一种将文本链式推理（CoR）整合到操作预测过程中的新型方法。我们通过此方法创建了TRACE数据集，这是一个大型集合，通过自主管道将指令与显式的文本推理关联起来。通过对这些数据进行微调，我们的模型学会了在行动前外部化其空间推理。我们的实验表明，我们的TRACE微调模型达到了最先进的性能，主Where2Place（W2P）基准上的准确率为48.1%（相对提高9.6%），更具有挑战性的W2P(h)子集上为55.0%。关键的是，消融研究证明性能直接与所使用的推理数据量相关，验证了链式推理的有效性。进一步分析模型的注意力图展示了可解释的推理过程，其中注意力在推理步骤之间动态转移。本研究展示了训练Vision-Language模型生成文本链式推理是一个有效且稳健的策略，以增强基于Vision-Language模型的机器人控制的精确性、可靠性和可解释性。我们的数据集和代码可在以下网址获得。 

---
# Vibe Learning: Education in the age of AI 

**Title (ZH)**: AI时代的学习：教育的变革 

**Authors**: Marcos Florencio, Francielle Prieto  

**Link**: [PDF](https://arxiv.org/pdf/2511.01956)  

**Abstract**: The debate over whether "thinking machines" could replace human intellectual labor has existed in both public and expert discussions since the mid-twentieth century, when the concept and terminology of Artificial Intelligence (AI) first emerged. For decades, this idea remained largely theoretical. However, with the recent advent of Generative AI - particularly Large Language Models (LLMs) - and the widespread adoption of tools such as ChatGPT, the issue has become a practical reality. Many fields that rely on human intellectual effort are now being reshaped by AI tools that both expand human capabilities and challenge the necessity of certain forms of work once deemed uniquely human but now easily automated. Education, somewhat unexpectedly, faces a pivotal responsibility: to devise long-term strategies for cultivating human skills that will remain relevant in an era of pervasive AI in the intellectual domain. In this context, we identify the limitations of current AI systems - especially those rooted in LLM technology - argue that the fundamental causes of these weaknesses cannot be resolved through existing methods, and propose directions within the constructivist paradigm for transforming education to preserve the long-term advantages of human intelligence over AI tools. 

**Abstract (ZH)**: 自20世纪中叶人工智能（AI）概念和术语首次出现以来，关于“思考机器”是否能取代人类智力劳动的辩论一直存在于公众和专家的讨论中。几十年来，这一想法主要停留在理论层面。然而，随着生成式AI（特别是大型语言模型LLMs）的兴起以及ChatGPT等工具的广泛采用，这一问题已成为现实。许多依赖人类智力劳动的领域现在正被既能扩展人类能力又能挑战某些形式工作的AI工具重塑。教育领域出乎意料地面临一个关键责任：制定长期策略以培养将在人工智能普及的时代仍然具有相关性的个人能力。在此背景下，我们识别当前AI系统的局限性，尤其是在大型语言模型技术领域，认为解决这些弱点的根本原因无法通过现有方法实现，并在建构主义 paradigm中提出转型教育的方向，以保持人类智能相对于AI工具的长期优势。 

---
# Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing 

**Title (ZH)**: 基于先验知识校准记忆探针的黑盒会员推理攻击针对低资源语言模型 

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01952)  

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at this https URL. 

**Abstract (ZH)**: 基于先验知识校准的大型视觉-语言模型黑箱会员推理框架 

---
# Interpretable Heart Disease Prediction via a Weighted Ensemble Model: A Large-Scale Study with SHAP and Surrogate Decision Trees 

**Title (ZH)**: 基于加权集成模型的可解释心脏疾病预测：大规模研究结合SHAP和代理决策树 

**Authors**: Md Abrar Hasnat, Md Jobayer, Md. Mehedi Hasan Shawon, Md. Golam Rabiul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2511.01947)  

**Abstract**: Cardiovascular disease (CVD) remains a critical global health concern, demanding reliable and interpretable predictive models for early risk assessment. This study presents a large-scale analysis using the Heart Disease Health Indicators Dataset, developing a strategically weighted ensemble model that combines tree-based methods (LightGBM, XGBoost) with a Convolutional Neural Network (CNN) to predict CVD risk. The model was trained on a preprocessed dataset of 229,781 patients where the inherent class imbalance was managed through strategic weighting and feature engineering enhanced the original 22 features to 25. The final ensemble achieves a statistically significant improvement over the best individual model, with a Test AUC of 0.8371 (p=0.003) and is particularly suited for screening with a high recall of 80.0%. To provide transparency and clinical interpretability, surrogate decision trees and SHapley Additive exPlanations (SHAP) are used. The proposed model delivers a combination of robust predictive performance and clinical transparency by blending diverse learning architectures and incorporating explainability through SHAP and surrogate decision trees, making it a strong candidate for real-world deployment in public health screening. 

**Abstract (ZH)**: 心血管疾病（CVD） remains a critical global health concern, demanding reliable and interpretable predictive models for early risk assessment. This study presents a large-scale analysis using the Heart Disease Health Indicators Dataset, developing a strategically weighted ensemble model that combines tree-based methods (LightGBM, XGBoost) with a Convolutional Neural Network (CNN) to predict CVD risk. The model was trained on a preprocessed dataset of 229,781 patients where the inherent class imbalance was managed through strategic weighting and feature engineering enhanced the original 22 features to 25. The final ensemble achieves a statistically significant improvement over the best individual model, with a Test AUC of 0.8371 (p=0.003) and is particularly suited for screening with a high recall of 80.0%. To provide transparency and clinical interpretability, surrogate decision trees and SHapley Additive exPlanations (SHAP) are used. The proposed model delivers a combination of robust predictive performance and clinical transparency by blending diverse learning architectures and incorporating explainability through SHAP and surrogate decision trees, making it a strong candidate for real-world deployment in public health screening. 

心血管疾病（CVD）仍然是全球健康的重要关切，需要可靠且可解释的预测模型来进行早期风险评估。本研究利用Heart Disease Health Indicators数据集进行大规模分析，开发了一个策略加权集成模型，将基于树的方法（LightGBM、XGBoost）与卷积神经网络（CNN）结合，以预测CVD风险。该模型在229,781例预处理患者数据集上进行训练，通过策略加权管理和特征工程处理，将原始的22个特征增强至25个，最终集成模型在个体模型上实现了统计显著性改进，测试AUC为0.8371（p=0.003），特别适用于筛查，召回率为80.0%。为了提供透明性和临床可解释性，使用了替代决策树和SHapley Additive exPlanations（SHAP）。所提模型通过融合多元学习架构并结合SHAP和替代决策树提供解释性，实现了稳健的预测性能和临床透明性，使其成为公共卫生筛查中实际部署的强有力候选模型。 

---
# COFAP: A Universal Framework for COFs Adsorption Prediction through Designed Multi-Modal Extraction and Cross-Modal Synergy 

**Title (ZH)**: COFAP：一种通过设计多模态提取和跨模态协同的通用COFs吸附预测框架 

**Authors**: Zihan Li, Mingyang Wan, Mingyu Gao, Zhongshan Chen, Xiangke Wang, Feifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01946)  

**Abstract**: Covalent organic frameworks (COFs) are promising adsorbents for gas adsorption and separation, while identifying the optimal structures among their vast design space requires efficient high-throughput screening. Conventional machine-learning predictors rely heavily on specific gas-related features. However, these features are time-consuming and limit scalability, leading to inefficiency and labor-intensive processes. Herein, a universal COFs adsorption prediction framework (COFAP) is proposed, which can extract multi-modal structural and chemical features through deep learning, and fuse these complementary features via cross-modal attention mechanism. Without Henry coefficients or adsorption heat, COFAP sets a new SOTA by outperforming previous approaches on hypoCOFs dataset. Based on COFAP, we also found that high-performing COFs for separation concentrate within a narrow range of pore size and surface area. A weight-adjustable prioritization scheme is also developed to enable flexible, application-specific ranking of candidate COFs for researchers. Superior efficiency and accuracy render COFAP directly deployable in crystalline porous materials. 

**Abstract (ZH)**: 共价有机框架（COFs）是气体吸附和分离的有前途的吸附剂，而在其广泛的设计空间中识别最优结构需要高效的高通量筛选。传统的机器学习预测器高度依赖于特定气体的相关特征。然而，这些特征的获取耗时且限制了可扩展性，导致效率低下和劳动密集型过程。在此，提出了一种通用的COFs吸附预测框架（COFAP），它可以通过深度学习提取多模式的结构和化学特征，并通过跨模态注意机制将这些互补特征融合。无需亨利系数或吸附热，COFAP在hypoCOFs数据集中超越了之前的方法，达到新的SOTA。基于COFAP，我们还发现高性能的COFs在分离时集中在狭窄的孔径和比表面积范围内。还开发了一种可调整权重的优先级方案，以使研究人员能够灵活地对候选COFs进行特定应用的排序。COFAP的卓越效率和准确性使其可以直接应用于结晶多孔材料中。 

---
# Detecting Vulnerabilities from Issue Reports for Internet-of-Things 

**Title (ZH)**: 从问题报告中检测物联网中的漏洞 

**Authors**: Sogol Masoumzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.01941)  

**Abstract**: Timely identification of issue reports reflecting software vulnerabilities is crucial, particularly for Internet-of-Things (IoT) where analysis is slower than non-IoT systems. While Machine Learning (ML) and Large Language Models (LLMs) detect vulnerability-indicating issues in non-IoT systems, their IoT use remains unexplored. We are the first to tackle this problem by proposing two approaches: (1) combining ML and LLMs with Natural Language Processing (NLP) techniques to detect vulnerability-indicating issues of 21 Eclipse IoT projects and (2) fine-tuning a pre-trained BERT Masked Language Model (MLM) on 11,000 GitHub issues for classifying \vul. Our best performance belongs to a Support Vector Machine (SVM) trained on BERT NLP features, achieving an Area Under the receiver operator characteristic Curve (AUC) of 0.65. The fine-tuned BERT achieves 0.26 accuracy, emphasizing the importance of exposing all data during training. Our contributions set the stage for accurately detecting IoT vulnerabilities from issue reports, similar to non-IoT systems. 

**Abstract (ZH)**: 及时识别反映软件漏洞的问题报告对于物联网（IoT）而言尤其重要，特别是当分析速度慢于非IoT系统时。尽管机器学习（ML）和大型语言模型（LLMs）可以检测非IoT系统中的漏洞指示问题，但它们在物联网中的应用尚未被探索。我们首次通过提出两种方法来解决这一问题：（1）结合机器学习、大型语言模型和自然语言处理（NLP）技术来检测21个Eclipse IoT项目的漏洞指示问题；（2）在11,000个GitHub问题上微调预训练的BERT遮蔽语言模型（MLM）以分类\vul。我们的最佳性能属于在BERT NLP特征上训练的支持向量机（SVM），其接收器操作特征曲线下的面积（AUC）为0.65。微调的BERT模型准确率为0.26，强调了训练过程中暴露所有数据的重要性。我们的贡献为从问题报告中精确检测物联网漏洞奠定了基础，类似于非IoT系统。 

---
# The Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold 

**Title (ZH)**: Grokking的几何学：零损失流形上的范数最小化 

**Authors**: Tiberiu Musat  

**Link**: [PDF](https://arxiv.org/pdf/2511.01938)  

**Abstract**: Grokking is a puzzling phenomenon in neural networks where full generalization occurs only after a substantial delay following the complete memorization of the training data. Previous research has linked this delayed generalization to representation learning driven by weight decay, but the precise underlying dynamics remain elusive. In this paper, we argue that post-memorization learning can be understood through the lens of constrained optimization: gradient descent effectively minimizes the weight norm on the zero-loss manifold. We formally prove this in the limit of infinitesimally small learning rates and weight decay coefficients. To further dissect this regime, we introduce an approximation that decouples the learning dynamics of a subset of parameters from the rest of the network. Applying this framework, we derive a closed-form expression for the post-memorization dynamics of the first layer in a two-layer network. Experiments confirm that simulating the training process using our predicted gradients reproduces both the delayed generalization and representation learning characteristic of grokking. 

**Abstract (ZH)**: Grokking现象中神经网络在完全记住训练数据后仅在一段时间延迟后实现全面泛化的机制可以通过约束优化的视角来理解：梯度下降在零损失流形上有效最小化权重范数。我们在学习率和权重衰减系数趋于无穷小的极限情况下形式上证明了这一点。为更深入地分析该现象，我们引入了一种近似方法，将一部分参数的学习动力学与其他网络部分分离。利用这一框架，我们推导出两层网络中第一层在后记忆化阶段的动力学的闭式表达式。实验结果证实，使用我们预测的梯度模拟训练过程可以重现grokking的延迟泛化和特征性表示学习。 

---
# Shorter but not Worse: Frugal Reasoning via Easy Samples as Length Regularizers in Math RLVR 

**Title (ZH)**: 短些但不逊色：通过易样本作为长度正则化器的节约推理 

**Authors**: Abdelaziz Bounhar, Hadi Abdine, Evan Dufraisse, Ahmad Chamma, Amr Mohamed, Dani Bouch, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01937)  

**Abstract**: Large language models (LLMs) trained for step-by-step reasoning often become excessively verbose, raising inference cost. Standard Reinforcement Learning with Verifiable Rewards (RLVR) pipelines filter out ``easy'' problems for training efficiency, leaving the model to train primarily on harder problems that require longer reasoning chains. This skews the output length distribution upward, resulting in a \textbf{model that conflates ``thinking longer'' with ``thinking better''}. In this work, we show that retaining and modestly up-weighting moderately easy problems acts as an implicit length regularizer. Exposing the model to solvable short-chain tasks constrains its output distribution and prevents runaway verbosity. The result is \textbf{\emph{emergent brevity for free}}: the model learns to solve harder problems without inflating the output length, \textbf{ despite the absence of any explicit length penalization}. RLVR experiments using this approach on \textit{Qwen3-4B-Thinking-2507} (with a 16k token limit) achieve baseline pass@1 AIME25 accuracy while generating solutions that are, on average, nearly twice as short. The code is available at \href{this https URL}{GitHub}, with datasets and models on \href{this https URL}{Hugging Face}. 

**Abstract (ZH)**: 大型语言模型（LLMs）训练用于逐步推理时往往会变得过度冗长，增加推理成本。标准可验证奖励强化学习（RLVR）管道过滤掉“简单”的问题以提高训练效率，使模型主要在需要较长推理链的较难问题上进行训练。这使得输出长度分布趋高，导致模型将“思考更长”与“思考更好”混为一谈。在本文中，我们展示了保留和适度增加中等难度问题作为隐式的长度正则化手段。使模型接触到可解决的短推理链任务可以约束其输出分布，防止过度冗长。结果是\emph{免费涌现的简明}: 模型能够在不增加输出长度的情况下学习解决更难的问题，\emph{即使没有明确的长度惩罚}。使用此方法在\textit{Qwen3-4B-Thinking-2507}（16k词令牌限制）上进行的RLVR实验实现了基线的AIME25准确率，同时生成的解决方案平均短近一倍。代码可在GitHub（\href{this https URL}{此链接}）上获取，数据集和模型可在Hugging Face（\href{this https URL}{此链接}）上获取。 

---
# Q-Sat AI: Machine Learning-Based Decision Support for Data Saturation in Qualitative Studies 

**Title (ZH)**: Q-Sat AI：基于机器学习的数据饱和度决策支持方法在定性研究中的应用 

**Authors**: Hasan Tutar, Caner Erden, Ümit Şentürk  

**Link**: [PDF](https://arxiv.org/pdf/2511.01935)  

**Abstract**: The determination of sample size in qualitative research has traditionally relied on the subjective and often ambiguous principle of data saturation, which can lead to inconsistencies and threaten methodological rigor. This study introduces a new, systematic model based on machine learning (ML) to make this process more objective. Utilizing a dataset derived from five fundamental qualitative research approaches - namely, Case Study, Grounded Theory, Phenomenology, Narrative Research, and Ethnographic Research - we developed an ensemble learning model. Ten critical parameters, including research scope, information power, and researcher competence, were evaluated using an ordinal scale and used as input features. After thorough preprocessing and outlier removal, multiple ML algorithms were trained and compared. The K-Nearest Neighbors (KNN), Gradient Boosting (GB), Random Forest (RF), XGBoost, and Decision Tree (DT) algorithms showed the highest explanatory power (Test R2 ~ 0.85), effectively modeling the complex, non-linear relationships involved in qualitative sampling decisions. Feature importance analysis confirmed the vital roles of research design type and information power, providing quantitative validation of key theoretical assumptions in qualitative methodology. The study concludes by proposing a conceptual framework for a web-based computational application designed to serve as a decision support system for qualitative researchers, journal reviewers, and thesis advisors. This model represents a significant step toward standardizing sample size justification, enhancing transparency, and strengthening the epistemological foundation of qualitative inquiry through evidence-based, systematic decision-making. 

**Abstract (ZH)**: 基于机器学习的质性研究样本大小确定的新系统模型 

---
# Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch 

**Title (ZH)**: Tool Zero: 从头基于纯强化学习训练工具增强的大语言模型 

**Authors**: Yirong Zeng, Xiao Ding, Yutai Hou, Yuxian Wang, Li Du, Juyi Dai, Qiuyang Ding, Duyu Tang, Dandan Tu, Weiwen Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01934)  

**Abstract**: Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods. 

**Abstract (ZH)**: 训练工具增强的大语言模型已成为提升语言模型处理复杂任务能力的有前途的方法。当前的监督微调范式依赖于构建广泛的专业领域数据集来训练模型。然而，这种方法往往难以有效地将模型推广到不熟悉或复杂的工具使用场景中。最近的强化学习（RL）范式能够赋予大语言模型更强的推理和泛化能力。在本工作中，我们解决了一个关键问题：纯粹的强化学习能否有效地激发模型的内在推理能力，并增强工具无关的泛化能力？我们提出了一种动态泛化引导的奖励设计，该设计基于规则的RL，逐步从探索性转向利用性工具使用模式。基于此设计，我们引入了Tool-Zero系列模型。这些模型通过直接从零模型（即未经后训练的基础模型）扩展RL来训练，使大语言模型能够自主利用通用工具。实验结果表明，在相同的实验设置下，我们模型的性能相比SFT和带有SFT的RL模型提高了超过7%。这些收益在跨数据集和同数据集评估中一致得到验证，验证了我们方法的有效性和稳健性。 

---
# Deciphering Personalization: Towards Fine-Grained Explainability in Natural Language for Personalized Image Generation Models 

**Title (ZH)**: 解析个性化：面向个性化图像生成模型的细粒度自然语言可解释性研究 

**Authors**: Haoming Wang, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01932)  

**Abstract**: Image generation models are usually personalized in practical uses in order to better meet the individual users' heterogeneous needs, but most personalized models lack explainability about how they are being personalized. Such explainability can be provided via visual features in generated images, but is difficult for human users to understand. Explainability in natural language is a better choice, but the existing approaches to explainability in natural language are limited to be coarse-grained. They are unable to precisely identify the multiple aspects of personalization, as well as the varying levels of personalization in each aspect. To address such limitation, in this paper we present a new technique, namely \textbf{FineXL}, towards \textbf{Fine}-grained e\textbf{X}plainability in natural \textbf{L}anguage for personalized image generation models. FineXL can provide natural language descriptions about each distinct aspect of personalization, along with quantitative scores indicating the level of each aspect of personalization. Experiment results show that FineXL can improve the accuracy of explainability by 56\%, when different personalization scenarios are applied to multiple types of image generation models. 

**Abstract (ZH)**: 精细化的自然语言解释技术FineXL：个性化图像生成模型的细粒度可解释性 

---
# Dynamic Population Distribution Aware Human Trajectory Generation with Diffusion Model 

**Title (ZH)**: 动态人口分布aware的人类轨迹生成方法——基于扩散模型 

**Authors**: Qingyue Long, Can Rong, Tong Li, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01929)  

**Abstract**: Human trajectory data is crucial in urban planning, traffic engineering, and public health. However, directly using real-world trajectory data often faces challenges such as privacy concerns, data acquisition costs, and data quality. A practical solution to these challenges is trajectory generation, a method developed to simulate human mobility behaviors. Existing trajectory generation methods mainly focus on capturing individual movement patterns but often overlook the influence of population distribution on trajectory generation. In reality, dynamic population distribution reflects changes in population density across different regions, significantly impacting individual mobility behavior. Thus, we propose a novel trajectory generation framework based on a diffusion model, which integrates the dynamic population distribution constraints to guide high-fidelity generation outcomes. Specifically, we construct a spatial graph to enhance the spatial correlation of trajectories. Then, we design a dynamic population distribution aware denoising network to capture the spatiotemporal dependencies of human mobility behavior as well as the impact of population distribution in the denoising process. Extensive experiments show that the trajectories generated by our model can resemble real-world trajectories in terms of some critical statistical metrics, outperforming state-of-the-art algorithms by over 54%. 

**Abstract (ZH)**: 人类轨迹数据在城市规划、交通工程和公共卫生中的应用至关重要。然而，直接使用真实世界轨迹数据常常面临着隐私顾虑、数据采集成本和数据质量等方面的挑战。一种实用的解决方案是轨迹生成，这种方法用于模拟人类移动行为。现有的轨迹生成方法主要侧重于捕捉个体移动模式，但往往忽略了人口分布对轨迹生成的影响。实际上，动态的人口分布反映了不同地区人口密度的变化，对个体移动行为产生了显著影响。因此，我们提出了一种基于扩散模型的新型轨迹生成框架，该框架整合了动态人口分布约束，以指导高保真生成结果。具体来说，我们构建了一个空间图来增强轨迹的空间相关性。然后，我们设计了一个动态人口分布感知的去噪网络，以捕捉人类移动行为的时空依赖性和去噪过程中的人口分布影响。大量的实验表明，由我们模型生成的轨迹在某些关键统计指标上与真实世界轨迹相似，并且在性能上超过了最先进的算法超过54%。 

---
# A Unified Model for Human Mobility Generation in Natural Disasters 

**Title (ZH)**: 自然灾难中人类移动性生成的统一模型 

**Authors**: Qingyue Long, Huandong Wang, Qi Ryan Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01928)  

**Abstract**: Human mobility generation in disaster scenarios plays a vital role in resource allocation, emergency response, and rescue coordination. During disasters such as wildfires and hurricanes, human mobility patterns often deviate from their normal states, which makes the task more challenging. However, existing works usually rely on limited data from a single city or specific disaster, significantly restricting the model's generalization capability in new scenarios. In fact, disasters are highly sudden and unpredictable, and any city may encounter new types of disasters without prior experience. Therefore, we aim to develop a one-for-all model for mobility generation that can generalize to new disaster scenarios. However, building a universal framework faces two key challenges: 1) the diversity of disaster types and 2) the heterogeneity among different cities. In this work, we propose a unified model for human mobility generation in natural disasters (named UniDisMob). To enable cross-disaster generalization, we design physics-informed prompt and physics-guided alignment that leverage the underlying common patterns in mobility changes after different disasters to guide the generation process. To achieve cross-city generalization, we introduce a meta-learning framework that extracts universal patterns across multiple cities through shared parameters and captures city-specific features via private parameters. Extensive experiments across multiple cities and disaster scenarios demonstrate that our method significantly outperforms state-of-the-art baselines, achieving an average performance improvement exceeding 13%. 

**Abstract (ZH)**: 自然灾害中的人群移动生成在资源分配、紧急响应和救援协调中发挥着至关重要的作用。在山火、飓风等灾难期间，人群移动模式往往与正常状态存在偏差，这使得任务更具挑战性。然而，现有工作通常依赖于单个城市或特定灾难的有限数据，大大限制了模型在新场景中的泛化能力。实际上，灾难高度突然且难以预测，任何城市都可能遭遇前所未有的新类型灾难。因此，我们旨在开发一个适用于所有场景的移动生成模型，能够泛化到新的灾难场景中。然而，构建通用框架面临两个关键挑战：1）灾难类型的多样性；2）不同城市之间的异质性。在本文中，我们提出了一种适用于自然灾难中人群移动生成的统一模型（命名为UniDisMob）。为实现跨灾难泛化，我们设计了物理启发式提示和物理引导对齐，利用不同灾难后移动变化的潜在共同模式来指导生成过程。为实现跨城市泛化，我们引入了元学习框架，通过共享参数提取多个城市的通用模式，同时通过私有参数捕捉特定城市的特性。跨多个城市和灾难场景的广泛实验表明，我们的方法显著优于现有最先进的基线方法，平均性能提升超过13%。 

---
# DeepContour: A Hybrid Deep Learning Framework for Accelerating Generalized Eigenvalue Problem Solving via Efficient Contour Design 

**Title (ZH)**: DeepContour：一种通过高效轮廓设计加速广义特征值问题求解的混合深度学习框架 

**Authors**: Yeqiu Chen, Ziyan Liu, Hong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01927)  

**Abstract**: Solving large-scale Generalized Eigenvalue Problems (GEPs) is a fundamental yet computationally prohibitive task in science and engineering. As a promising direction, contour integral (CI) methods, such as the CIRR algorithm, offer an efficient and parallelizable framework. However, their performance is critically dependent on the selection of integration contours -- improper selection without reliable prior knowledge of eigenvalue distribution can incur significant computational overhead and compromise numerical accuracy. To address this challenge, we propose DeepContour, a novel hybrid framework that integrates a deep learning-based spectral predictor with Kernel Density Estimation for principled contour design. Specifically, DeepContour first employs a Fourier Neural Operator (FNO) to rapidly predict the spectral distribution of a given GEP. Subsequently, Kernel Density Estimation (KDE) is applied to the predicted spectrum to automatically and systematically determine proper integration contours. Finally, these optimized contours guide the CI solver to efficiently find the desired eigenvalues. We demonstrate the effectiveness of our method on diverse challenging scientific problems. In our main experiments, DeepContour accelerates GEP solving across multiple datasets, achieving up to a 5.63$\times$ speedup. By combining the predictive power of deep learning with the numerical rigor of classical solvers, this work pioneers an efficient and robust paradigm for tackling difficult generalized eigenvalue involving matrices of high dimension. 

**Abstract (ZH)**: 大规模广义特征值问题（GEPs）的求解是科学和工程领域一个基本但计算上极具挑战性的任务。作为有前景的方向，曲率积分（CI）方法，如CIRR算法，提供了高效且并行可编程的框架。然而，其性能严重依赖于积分轮廓的选择——如果没有可靠的特征值分布先验知识，不当的选择会导致显著的计算开销并损害数值精度。为解决这一挑战，我们提出了一种名为DeepContour的新型混合框架，它将基于深度学习的谱预测器与核密度估计相结合，实现原理性的轮廓设计。具体来说，DeepContour首先使用傅里叶神经算子（FNO）快速预测给定GEP的谱分布。随后，应用核密度估计（KDE）到预测的谱中，以自动且系统地确定合适的积分轮廓。最后，优化的轮廓引导CI求解器高效地找到所需的特征值。我们在多种挑战性的科学问题上展示了我们方法的有效性。在主要实验中，DeepContour在多个数据集上加速了广义特征值问题的求解，最高加速倍数可达5.63倍。通过结合深度学习的预测能力和经典求解器的数值严谨性，本文开创了一种处理高维矩阵涉及的困难广义特征值问题的有效且稳健范式。 

---
# Neural Green's Functions 

**Title (ZH)**: 神经格林函数 

**Authors**: Seungwoo Yoo, Kyeongmin Yeo, Jisung Hwang, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2511.01924)  

**Abstract**: We introduce Neural Green's Function, a neural solution operator for linear partial differential equations (PDEs) whose differential operators admit eigendecompositions. Inspired by Green's functions, the solution operators of linear PDEs that depend exclusively on the domain geometry, we design Neural Green's Function to imitate their behavior, achieving superior generalization across diverse irregular geometries and source and boundary functions. Specifically, Neural Green's Function extracts per-point features from a volumetric point cloud representing the problem domain and uses them to predict a decomposition of the solution operator, which is subsequently applied to evaluate solutions via numerical integration. Unlike recent learning-based solution operators, which often struggle to generalize to unseen source or boundary functions, our framework is, by design, agnostic to the specific functions used during training, enabling robust and efficient generalization. In the steady-state thermal analysis of mechanical part geometries from the MCB dataset, Neural Green's Function outperforms state-of-the-art neural operators, achieving an average error reduction of 13.9\% across five shape categories, while being up to 350 times faster than a numerical solver that requires computationally expensive meshing. 

**Abstract (ZH)**: 基于神经网络的Green函数：一种适用于具有特征分解的线性偏微分方程的神经求解算子 

---
# Fibbinary-Based Compression and Quantization for Efficient Neural Radio Receivers 

**Title (ZH)**: 基于Fibbinary的压缩与量化方法在高效神经射频接收机中的应用 

**Authors**: Roberta Fiandaca, Manil Dev Gomony  

**Link**: [PDF](https://arxiv.org/pdf/2511.01921)  

**Abstract**: Neural receivers have shown outstanding performance compared to the conventional ones but this comes with a high network complexity leading to a heavy computational cost. This poses significant challenges in their deployment on hardware-constrained devices. To address the issue, this paper explores two optimization strategies: quantization and compression. We introduce both uniform and non-uniform quantization such as the Fibonacci Code word Quantization (FCQ). A novel fine-grained approach to the Incremental Network Quantization (INQ) strategy is then proposed to compensate for the losses introduced by the above mentioned quantization techniques. Additionally, we introduce two novel lossless compression algorithms that effectively reduce the memory size by compressing sequences of Fibonacci quantized parameters characterized by a huge redundancy. The quantization technique provides a saving of 45\% and 44\% in the multiplier's power and area, respectively, and its combination with the compression determines a 63.4\% reduction in memory footprint, while still providing higher performances than a conventional receiver. 

**Abstract (ZH)**: 神经接收机展示了与传统接收机相比出色的表现，但同时带来了高网络复杂性，导致高昂的计算成本。这在硬件受限设备上部署时提出了重大挑战。为解决该问题，本文探索了两种优化策略：量化和压缩。介绍了均匀量化和非均匀量化，包括Fibonacci码字量化（FCQ）。提出了一种对增量网络量化（INQ）策略的新型精细粒度方法，以补偿上述量化技术引入的损失。此外，引入了两种新的无损压缩算法，有效减少了由大量冗余导致的Fibonacci量化参数序列占用的内存空间。量化技术在乘法器功耗和面积上分别节省了45%和44%，与压缩结合使用时，内存占用减少了63.4%，同时提供了高于传统接收机的性能。 

---
# iFlyBot-VLA Technical Report 

**Title (ZH)**: iFlyBot-VLA 技术报告 

**Authors**: Yuan Zhang, Chenyu Xue, Wenjie Xu, Chao Ji, Jiajia wu, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01914)  

**Abstract**: We introduce iFlyBot-VLA, a large-scale Vision-Language-Action (VLA) model trained under a novel framework. The main contributions are listed as follows: (1) a latent action model thoroughly trained on large-scale human and robotic manipulation videos; (2) a dual-level action representation framework that jointly supervises both the Vision-Language Model (VLM) and the action expert during training; (3) a mixed training strategy that combines robot trajectory data with general QA and spatial QA datasets, effectively enhancing the 3D perceptual and reasoning capabilities of the VLM backbone. Specifically, the VLM is trained to predict two complementary forms of actions: latent actions, derived from our latent action model pretrained on cross-embodiment manipulation data, which capture implicit high-level intentions; and structured discrete action tokens, obtained through frequency-domain transformations of continuous control signals, which encode explicit low-level dynamics. This dual supervision aligns the representation spaces of language, vision, and action, enabling the VLM to directly contribute to action generation. Experimental results on the LIBERO Franka benchmark demonstrate the superiority of our frame-work, while real-world evaluations further show that iFlyBot-VLA achieves competitive success rates across diverse and challenging manipulation tasks. Furthermore, we plan to open-source a portion of our self-constructed dataset to support future research in the community 

**Abstract (ZH)**: 我们介绍了一种新型框架下的大规模视觉-语言-动作（VLA）模型iFlyBot-VLA。主要贡献如下：(1) 一种在大规模人类和机器人操作视频上充分训练的潜在动作模型；(2) 一种双层动作表示框架，在训练过程中同时监督视觉语言模型（VLM）和动作专家；(3) 一种结合机器人轨迹数据和通用问答及空间问答数据的混合训练策略，有效增强VLM主干网络的3D感知和推理能力。具体而言，VLM被训练预测两种互补形式的动作：从跨主体操作数据预训练的潜在动作模型中提取的潜在动作，捕捉隐含的高层意图；以及通过连续控制信号的频域变换获得的结构离散动作标记，编码显式的低层动态。这种双监督使语言、视觉和动作的表示空间对齐，从而使VLM能够直接参与到动作生成中。在LIBERO Franka基准上的实验结果证明了我们框架的优势，而现实世界的评估进一步表明，iFlyBot-VLA在各种具有挑战性的操作任务中取得了竞争力的表现。此外，我们将开放部分自主构建的数据集以支持社区的未来研究。 

---
# EvoMem: Improving Multi-Agent Planning with Dual-Evolving Memory 

**Title (ZH)**: EvoMem: 通过双演化记忆提高多agent规划能力 

**Authors**: Wenzhe Fan, Ning Yan, Masood Mortazavi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01912)  

**Abstract**: Planning has been a cornerstone of artificial intelligence for solving complex problems, and recent progress in LLM-based multi-agent frameworks have begun to extend this capability. However, the role of human-like memory within these frameworks remains largely unexplored. Understanding how agents coordinate through memory is critical for natural language planning, where iterative reasoning, constraint tracking, and error correction drive the success. Inspired by working memory model in cognitive psychology, we present EvoMem, a multi-agent framework built on a dual-evolving memory mechanism. The framework consists of three agents (Constraint Extractor, Verifier, and Actor) and two memory modules: Constraint Memory (CMem), which evolves across queries by storing task-specific rules and constraints while remains fixed within a query, and Query-feedback Memory (QMem), which evolves within a query by accumulating feedback across iterations for solution refinement. Both memory modules are reset at the end of each query session. Evaluations on trip planning, meeting planning, and calendar scheduling show consistent performance improvements, highlighting the effectiveness of EvoMem. This success underscores the importance of memory in enhancing multi-agent planning. 

**Abstract (ZH)**: 基于演化记忆的多智能体规划框架 

---
# Variational Geometry-aware Neural Network based Method for Solving High-dimensional Diffeomorphic Mapping Problems 

**Title (ZH)**: 基于变分几何感知神经网络的方法求解高维非欧几里得映射问题 

**Authors**: Zhiwen Li, Cheuk Hin Ho, Lok Ming Lui  

**Link**: [PDF](https://arxiv.org/pdf/2511.01911)  

**Abstract**: Traditional methods for high-dimensional diffeomorphic mapping often struggle with the curse of dimensionality. We propose a mesh-free learning framework designed for $n$-dimensional mapping problems, seamlessly combining variational principles with quasi-conformal theory. Our approach ensures accurate, bijective mappings by regulating conformality distortion and volume distortion, enabling robust control over deformation quality. The framework is inherently compatible with gradient-based optimization and neural network architectures, making it highly flexible and scalable to higher-dimensional settings. Numerical experiments on both synthetic and real-world medical image data validate the accuracy, robustness, and effectiveness of the proposed method in complex registration scenarios. 

**Abstract (ZH)**: 高维度差分同胚映射的传统方法往往难以应对维度灾难。我们提出了一种无网格学习框架，用于解决$n$维映射问题，该框架无缝结合了变分原理与准共形理论。通过调节共形失真和体积失真，该方法确保了映射的精确性和双射性，从而能够 robust 地控制变形质量。该框架固有的与基于梯度的优化和神经网络架构兼容性，使其在高维设置中具有高度的灵活性和可扩展性。数值实验在合成和真实世界医学图像数据上的结果验证了所提出方法在复杂配准场景下的准确性和有效性。 

---
# Between Myths and Metaphors: Rethinking LLMs for SRH in Conservative Contexts 

**Title (ZH)**: 在神话与隐喻之间：重新思考保守背景下的人工智能语言模型在性与生殖健康领域的应用 

**Authors**: Ameemah Humayun, Bushra Zubair, Maryam Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2511.01907)  

**Abstract**: Low-resource countries represent over 90% of maternal deaths, with Pakistan among the top four countries contributing nearly half in 2023. Since these deaths are mostly preventable, large language models (LLMs) can help address this crisis by automating health communication and risk assessment. However, sexual and reproductive health (SRH) communication in conservative contexts often relies on indirect language that obscures meaning, complicating LLM-based interventions. We conduct a two-stage study in Pakistan: (1) analyzing data from clinical observations, interviews, and focus groups with clinicians and patients, and (2) evaluating the interpretive capabilities of five popular LLMs on this data. Our analysis identifies two axes of communication (referential domain and expression approach) and shows LLMs struggle with semantic drift, myths, and polysemy in clinical interactions. We contribute: (1) empirical themes in SRH communication, (2) a categorization framework for indirect communication, (3) evaluation of LLM performance, and (4) design recommendations for culturally-situated SRH communication. 

**Abstract (ZH)**: 低资源国家代表了超过90%的 maternal死亡，其中巴基斯坦在2023年贡献了近半数。由于这些死亡主要是可以预防的，大规模语言模型（LLMs）可以通过自动化健康沟通和风险评估来帮助应对这一危机。然而，在保守的背景下，性与生殖健康（SRH）沟通往往依赖于间接语言，这使得基于LLM的干预措施复杂化。我们在巴基斯坦进行了两阶段研究：（1）分析临床观察、访谈和 clinicians及患者焦点小组的数据，（2）评估五种流行LLM在这方面的解释能力。我们的分析确定了沟通的两个维度（指称领域和表达方式），并表明LLMs在临床互动中面临语义转移、神话和多义性的挑战。我们贡献了：（1）SRH沟通的实证主题，（2）间接沟通的分类框架，（3）LLM性能评估，以及（4）基于文化背景的SRH沟通设计建议。 

---
# Thinking Like a Student: AI-Supported Reflective Planning in a Theory-Intensive Computer Science Course 

**Title (ZH)**: 从学生视角思考：AI支持的反思性规划在理论密集型计算机科学课程中的应用 

**Authors**: Noa Izsak  

**Link**: [PDF](https://arxiv.org/pdf/2511.01906)  

**Abstract**: In the aftermath of COVID-19, many universities implemented supplementary "reinforcement" roles to support students in demanding courses. Although the name for such roles may differ between institutions, the underlying idea of providing structured supplementary support is common. However, these roles were often poorly defined, lacking structured materials, pedagogical oversight, and integration with the core teaching team. This paper reports on the redesign of reinforcement sessions in a challenging undergraduate course on formal methods and computational models, using a large language model (LLM) as a reflective planning tool. The LLM was prompted to simulate the perspective of a second-year student, enabling the identification of conceptual bottlenecks, gaps in intuition, and likely reasoning breakdowns before classroom delivery. These insights informed a structured, repeatable session format combining targeted review, collaborative examples, independent student work, and guided walkthroughs. Conducted over a single semester, the intervention received positive student feedback, indicating increased confidence, reduced anxiety, and improved clarity, particularly in abstract topics such as the pumping lemma and formal language expressive power comparisons. The findings suggest that reflective, instructor-facing use of LLMs can enhance pedagogical design in theoretically dense domains and may be adaptable to other cognitively demanding computer science courses. 

**Abstract (ZH)**: COVID-19之后，在一门形式方法与计算模型的挑战性本科课程中重新设计强化会话：使用大型语言模型作为反思性规划工具 

---
# Before the Clinic: Transparent and Operable Design Principles for Healthcare AI 

**Title (ZH)**: clinic之前:透明可操作的医疗AI设计原则 

**Authors**: Alexander Bakumenko, Aaron J. Masino, Janine Hoelscher  

**Link**: [PDF](https://arxiv.org/pdf/2511.01902)  

**Abstract**: The translation of artificial intelligence (AI) systems into clinical practice requires bridging fundamental gaps between explainable AI theory, clinician expectations, and governance requirements. While conceptual frameworks define what constitutes explainable AI (XAI) and qualitative studies identify clinician needs, little practical guidance exists for development teams to prepare AI systems prior to clinical evaluation. We propose two foundational design principles, Transparent Design and Operable Design, that operationalize pre-clinical technical requirements for healthcare AI. Transparent Design encompasses interpretability and understandability artifacts that enable case-level reasoning and system traceability. Operable Design encompasses calibration, uncertainty, and robustness to ensure reliable, predictable system behavior under real-world conditions. We ground these principles in established XAI frameworks, map them to documented clinician needs, and demonstrate their alignment with emerging governance requirements. This pre-clinical playbook provides actionable guidance for development teams, accelerates the path to clinical evaluation, and establishes a shared vocabulary bridging AI researchers, healthcare practitioners, and regulatory stakeholders. By explicitly scoping what can be built and verified before clinical deployment, we aim to reduce friction in clinical AI translation while remaining cautious about what constitutes validated, deployed explainability. 

**Abstract (ZH)**: 将人工智能系统应用于临床实践需要弥合可解释人工智能理论、 clinicians 期望与治理要求之间的基本差距。我们提出了两种基础设计原则：透明设计和可操作设计，以实现医疗人工智能的临床前期技术要求。透明设计涵盖了用于案例级推理和系统可追溯性的可解释性和可理解性 artifacts。可操作设计涵盖了校准、不确定性以及鲁棒性，以确保在实际条件下的可靠和可预测系统行为。这些原则以现有的可解释人工智能框架为基础，映射到记录的 clinician 需求，并展示了与新兴治理要求的一致性。此临床前期 playbook 为开发团队提供可操作的指导，加速临床评估的路径，并建立一种共享语言以连接人工智能研究人员、医疗实践者和监管利益相关者。通过明确界定在临床部署前可以构建和验证的内容，我们旨在减少临床人工智能翻译中的摩擦，同时谨慎对待验证和部署的解释性问题。 

---
# LGCC: Enhancing Flow Matching Based Text-Guided Image Editing with Local Gaussian Coupling and Context Consistency 

**Title (ZH)**: LGCC：基于局部高斯耦合和上下文一致性的情感流匹配驱动文本指导图像编辑 

**Authors**: Fangbing Liu, Pengfei Duan, Wen Li, Yi He  

**Link**: [PDF](https://arxiv.org/pdf/2511.01894)  

**Abstract**: Recent advancements have demonstrated the great potential of flow matching-based Multimodal Large Language Models (MLLMs) in image editing. However, state-of-the-art works like BAGEL face limitations, including detail degradation, content inconsistency, and inefficiency due to their reliance on random noise initialization. To address these issues, we propose LGCC, a novel framework with two key components: Local Gaussian Noise Coupling (LGNC) and Content Consistency Loss (CCL). LGNC preserves spatial details by modeling target image embeddings and their locally perturbed counterparts as coupled pairs, while CCL ensures semantic alignment between edit instructions and image modifications, preventing unintended content removal. By integrating LGCC with the BAGEL pre-trained model via curriculum learning, we significantly reduce inference steps, improving local detail scores on I2EBench by 1.60% and overall scores by 0.53%. LGCC achieves 3x -- 5x speedup for lightweight editing and 2x for universal editing, requiring only 40% -- 50% of the inference time of BAGEL or Flux. These results demonstrate LGCC's ability to preserve detail, maintain contextual integrity, and enhance inference speed, offering a cost-efficient solution without compromising editing quality. 

**Abstract (ZH)**: Recent Advancements Have Demonstrated the Great Potential of Flow Matching-Based Multimodal Large Language Models (MLLMs) in Image Editing: Addressing Limitations with LGCC, a Novel Framework 

---
# Multi-Personality Generation of LLMs at Decoding-time 

**Title (ZH)**: 解码时LLM的多个性格生成 

**Authors**: Rongxin Chen, Yunfan Li, Yige Yuan, Bingbing Xu, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01891)  

**Abstract**: Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at this https URL . 

**Abstract (ZH)**: 多个性格生成框架：在解码时结合多个个性特征，对于大型语言模型来说是一项基础性挑战。现有的基于重新训练的方法成本高且扩展性差，而解码时的方法通常依赖于外部模型或启发式方法，限制了灵活性和鲁棒性。在本文中，我们提出了一种新颖的多个性格生成（MPG）框架，该框架在解码时结合多个个性特征，无需依赖稀缺的多维模型或额外训练，而是利用单维模型中的隐含密度比来重新定义任务为从这些比率聚合的目标策略中采样。为了高效实施MPG，我们设计了推测性块级拒绝采样（SCR），该方法分块生成响应并在滑动窗口内并行验证它们，这显著减少了计算开销同时保持高质量生成。实验结果表明，MPG在MBTI人格和角色扮演任务中表现出色，生成质量可提升16%-18%。代码和数据可在该网址获取。 

---
# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization 

**Title (ZH)**: CudaForge: 一种带有硬件反馈的CUDA内核优化智能体框架 

**Authors**: Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.01884)  

**Abstract**: Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at this https URL 

**Abstract (ZH)**: 基于CUDA内核生成与优化的无训练多智能体工作流CudaForge 

---
# CytoNet: A Foundation Model for the Human Cerebral Cortex 

**Title (ZH)**: CytoNet: 人类大脑皮层的基础模型 

**Authors**: Christian Schiffer, Zeynep Boztoprak, Jan-Oliver Kropp, Julia Thönnißen, Katia Berr, Hannah Spitzer, Katrin Amunts, Timo Dickscheid  

**Link**: [PDF](https://arxiv.org/pdf/2511.01870)  

**Abstract**: To study how the human brain works, we need to explore the organization of the cerebral cortex and its detailed cellular architecture. We introduce CytoNet, a foundation model that encodes high-resolution microscopic image patches of the cerebral cortex into highly expressive feature representations, enabling comprehensive brain analyses. CytoNet employs self-supervised learning using spatial proximity as a powerful training signal, without requiring manual labelling. The resulting features are anatomically sound and biologically relevant. They encode general aspects of cortical architecture and unique brain-specific traits. We demonstrate top-tier performance in tasks such as cortical area classification, cortical layer segmentation, cell morphology estimation, and unsupervised brain region mapping. As a foundation model, CytoNet offers a consistent framework for studying cortical microarchitecture, supporting analyses of its relationship with other structural and functional brain features, and paving the way for diverse neuroscientific investigations. 

**Abstract (ZH)**: 为了研究人类大脑的工作原理，我们需要探讨大脑皮层的组织结构及其详细的细胞架构。我们 introduces CytoNet，一种基础模型，将大脑皮层的高分辨率显微图像片段编码为高度表达的特征表示，从而实现全面的大脑分析。CytoNet 使用基于空间接近性的自监督学习进行训练，无需手动标注。生成的特征具有解剖学和生物学的相关性，能够编码皮层架构的一般特征和独特的脑部特异性特征。我们在皮层区域分类、皮层层段分割、细胞形态估计和无监督脑区映射等任务中展示了顶级性能。作为基础模型，CytoNet 提供了一致的框架来研究皮层微架构，支持其与其他结构和功能脑特征的关系分析，并为多样的神经科学调查铺平了道路。 

---
# DiffPace: Diffusion-based Plug-and-play Augmented Channel Estimation in mmWave and Terahertz Ultra-Massive MIMO Systems 

**Title (ZH)**: DiffPace: 基于扩散的插件式超密集信道估计方法在毫米波和太赫兹超大规模MIMO系统中 

**Authors**: Zhengdong Hu, Chong Han, Wolfgang Gerstacker, Robert Schober  

**Link**: [PDF](https://arxiv.org/pdf/2511.01867)  

**Abstract**: Millimeter-wave (mmWave) and Terahertz (THz)-band communications hold great promise in meeting the growing data-rate demands of next-generation wireless networks, offering abundant bandwidth. To mitigate the severe path loss inherent to these high frequencies and reduce hardware costs, ultra-massive multiple-input multiple-output (UM-MIMO) systems with hybrid beamforming architectures can deliver substantial beamforming gains and enhanced spectral efficiency. However, accurate channel estimation (CE) in mmWave and THz UM-MIMO systems is challenging due to high channel dimensionality and compressed observations from a limited number of RF chains, while the hybrid near- and far-field radiation patterns, arising from large array apertures and high carrier frequencies, further complicate CE. Conventional compressive sensing based frameworks rely on predefined sparsifying matrices, which cannot faithfully capture the hybrid near-field and far-field channel structures, leading to degraded estimation performance. This paper introduces DiffPace, a diffusion-based plug-and-play method for channel estimation. DiffPace uses a diffusion model (DM) to capture the channel distribution based on the hybrid spherical and planar-wave (HPSM) model. By applying the plug-and-play approach, it leverages the DM as prior knowledge, improving CE accuracy. Moreover, DM performs inference by solving an ordinary differential equation, minimizing the number of required inference steps compared with stochastic sampling method. Experimental results show that DiffPace achieves competitive CE performance, attaining -15 dB normalized mean square error (NMSE) at a signal-to-noise ratio (SNR) of 10 dB, with 90\% fewer inference steps compared to state-of-the-art schemes, simultaneously providing high estimation precision and enhanced computational efficiency. 

**Abstract (ZH)**: 基于扩散的插件式通道估计方法DiffPace 

---
# EdgeReasoning: Characterizing Reasoning LLM Deployment on Edge GPUs 

**Title (ZH)**: 边缘推理：边端GPU上推理预训练语言模型的特点研究 

**Authors**: Benjamin Kubwimana, Qijing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01866)  

**Abstract**: Edge intelligence paradigm is increasingly demanded by the emerging autonomous systems, such as robotics. Beyond ensuring privacy-preserving operation and resilience in connectivity-limited environments, edge deployment offers significant energy and cost advantages over cloud-based solutions. However, deploying large language models (LLMs) for reasoning tasks on edge GPUs faces critical challenges from strict latency constraints and limited computational resources. To navigate these constraints, developers must balance multiple design factors - choosing reasoning versus non-reasoning architectures, selecting appropriate model sizes, allocating token budgets, and applying test-time scaling strategies - to meet target latency and optimize accuracy. Yet guidance on optimal combinations of these variables remains scarce. In this work, we present EdgeReasoning, a comprehensive study characterizing the deployment of reasoning LLMs on edge GPUs. We systematically quantify latency-accuracy tradeoffs across various LLM architectures and model sizes. We systematically evaluate prompt-based and model-tuning-based techniques for reducing reasoning token length while maintaining performance quality. We further profile test-time scaling methods with varying degrees of parallelism to maximize accuracy under strict latency budgets. Through these analyses, EdgeReasoning maps the Pareto frontier of achievable accuracy-latency configurations, offering systematic guidance for optimal edge deployment of reasoning LLMs. 

**Abstract (ZH)**: 边缘智能范式日益被新兴自主系统，如机器人所需求。除了在连接受限环境中确保隐私保护操作和弹性之外，边缘部署在能耗和成本方面相比基于云的解决方案具有显著优势。然而，在边缘GPU上部署大型语言模型（LLMs）进行推理任务面临着严格延迟约束和有限计算资源的关键挑战。为了应对这些限制，开发者必须在多重设计因素之间进行平衡——选择推理架构还是非推理架构、选择合适模型规模、分配令牌预算，并应用测试时缩放策略，以满足延迟目标并优化准确性。然而，关于这些变量的最优组合的指导仍然稀缺。在本工作中，我们提出了EdgeReasoning，一项全面研究边缘GPU上部署推理LLMs的特性。我们系统地量化了各种LLM架构和模型规模下的延迟-准确性权衡。我们系统地评估了基于提示和基于模型调优的技术，以减少推理令牌长度同时保持性能质量。我们进一步分析了具有不同并行度的测试时缩放方法，以在严格延迟预算下最大化准确性。通过这些分析，EdgeReasoning绘制了可实现的准确性-延迟配置的帕累托前沿，为推理LLMs的最优边缘部署提供系统性指导。 

---
