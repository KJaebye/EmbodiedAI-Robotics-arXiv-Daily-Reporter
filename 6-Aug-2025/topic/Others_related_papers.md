# LiGen: GAN-Augmented Spectral Fingerprinting for Indoor Positioning 

**Title (ZH)**: LiGen: 基于GAN增强的光谱指纹定位方法 

**Authors**: Jie Lin, Hsun-Yu Lee, Ho-Ming Li, Fang-Jing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03024)  

**Abstract**: Accurate and robust indoor localization is critical for smart building applications, yet existing Wi-Fi-based systems are often vulnerable to environmental conditions. This work presents a novel indoor localization system, called LiGen, that leverages the spectral intensity patterns of ambient light as fingerprints, offering a more stable and infrastructure-free alternative to radio signals. To address the limited spectral data, we design a data augmentation framework based on generative adversarial networks (GANs), featuring two variants: PointGAN, which generates fingerprints conditioned on coordinates, and FreeGAN, which uses a weak localization model to label unconditioned samples. Our positioning model, leveraging a Multi-Layer Perceptron (MLP) architecture to train on synthesized data, achieves submeter-level accuracy, outperforming Wi-Fi-based baselines by over 50\%. LiGen also demonstrates strong robustness in cluttered environments. To the best of our knowledge, this is the first system to combine spectral fingerprints with GAN-based data augmentation for indoor localization. 

**Abstract (ZH)**: 基于光谱强度模式的鲁棒室内定位系统LiGen 

---
# Beyond Policy Optimization: A Data Curation Flywheel for Sparse-Reward Long-Horizon Planning 

**Title (ZH)**: 超越策略优化：稀疏奖励长时规划的数据整理飞轮 

**Authors**: Yutong Wang, Pengliang Ji, Kaixin Li, Baolong Bi, Tao Feng, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2508.03018)  

**Abstract**: Large Language Reasoning Models have demonstrated remarkable success on static tasks, yet their application to multi-round agentic planning in interactive environments faces two fundamental challenges. First, the intractable credit assignment problem renders conventional reinforcement learning ineffective in sparse-reward settings. Second, the computational overhead of verbose, step-by-step reasoning histories is prohibitive. To address these challenges, we propose BPO, a three-stage framework (bootstrapping, extrapolation, and refinement) that establishes a self-improving data flywheel to develop robust reasoning models for long-horizon, sparse-reward environments. Our framework first bootstraps efficient reasoning using the proposed planning quaternions with long-short chain-of-thought fusion. It then extrapolates to out-of-distribution tasks through complexity-stratified curriculum learning. Finally, the model iteratively refines itself by learning exclusively on experiences selected via reward-gated rejection sampling. Experiments on ALFWorld, ScienceWorld, and WebShop demonstrate that our approach achieves state-of-the-art with significant token efficiency, providing a new recipe for reasoning models in agentic planning. 

**Abstract (ZH)**: 大型语言推理模型在静态任务上取得了显著成功，但在交互环境中应用于多轮代理规划面临两个根本挑战。为应对这些挑战，我们提出了BPO框架（自增强、外推和精炼三个阶段），该框架建立了一种自我改进的数据飞轮，旨在开发适用于长期决策、稀疏奖励环境的 robust 推理模型。该框架首先利用提出的融合长短期思考的规划四元数进行高效推理，然后通过分层课程学习来外推到分布外任务，最后通过奖励门控拒绝采样选择的经验进行迭代精炼。我们在ALFWorld、ScienceWorld和WebShop上的实验表明，该方法在显著提高 token 效率的同时达到了最先进的效果，为代理规划中的推理模型提供了一种新的配方。 

---
# Hidden Dynamics of Massive Activations in Transformer Training 

**Title (ZH)**: Transformer训练中大规模激活的隐式动态 

**Authors**: Jorge Gallego-Feliciano, S. Aaron McClendon, Juan Morinelli, Stavros Zervoudakis, Antonios Saravanos  

**Link**: [PDF](https://arxiv.org/pdf/2508.03616)  

**Abstract**: Massive activations are scalar values in transformer hidden states that achieve values orders of magnitude larger than typical activations and have been shown to be critical for model functionality. While prior work has characterized these phenomena in fully trained models, the temporal dynamics of their emergence during training remain poorly understood. We present the first comprehensive analysis of massive activation development throughout transformer training, using the Pythia model family as our testbed. Through systematic analysis of various model sizes across multiple training checkpoints, we demonstrate that massive activation emergence follows predictable mathematical patterns that can be accurately modeled using an exponentially-modulated logarithmic function with five key parameters. We develop a machine learning framework to predict these mathematical parameters from architectural specifications alone, achieving high accuracy for steady-state behavior and moderate accuracy for emergence timing and magnitude. These findings enable architects to predict and potentially control key aspects of massive activation emergence through design choices, with significant implications for model stability, training cycle length, interpretability, and optimization. Our findings demonstrate that the emergence of massive activations is governed by model design and can be anticipated, and potentially controlled, before training begins. 

**Abstract (ZH)**: 大规模激活是变压器隐藏状态中的标量值，其取值比典型激活高出几个数量级，并且已被证明对模型功能至关重要。尽管先前的工作已经在完全训练的模型中表征了这些现象，但在训练过程中这些现象出现的时序动态仍然知之甚少。我们首次全面分析了变压器训练过程中大规模激活的发展情况，以Pythia模型家族作为测试平台。通过系统分析各种模型大小在多个训练检查点上的表现，我们证明了大规模激活的出现遵循可预测的数学模式，并可以使用五种子参数的指数调制对数函数进行准确建模。我们开发了一种机器学习框架，仅从架构规格中预测这些数学参数，实现了对稳定态行为的高精度预测，并对出现时机和程度实现了中等精度的预测。这些发现使架构师能够通过设计选择预测并潜在地控制大规模激活出现的关键方面，从而对模型的稳定性和训练周期长度、可解释性和优化产生重大影响。我们的研究结果表明，大规模激活的出现受模型设计的控制，并可以在开始训练之前进行预测和潜在控制。 

---
# Toward a Graph-Theoretic Model of Belief: Confidence, Credibility, and Structural Coherence 

**Title (ZH)**: 基于图论的信念模型：信心、可信度与结构一致性 

**Authors**: Saleh Nikooroo  

**Link**: [PDF](https://arxiv.org/pdf/2508.03465)  

**Abstract**: Belief systems are often treated as globally consistent sets of propositions or as scalar-valued probability distributions. Such representations tend to obscure the internal structure of belief, conflate external credibility with internal coherence, and preclude the modeling of fragmented or contradictory epistemic states. This paper introduces a minimal formalism for belief systems as directed, weighted graphs. In this framework, nodes represent individual beliefs, edges encode epistemic relationships (e.g., support or contradiction), and two distinct functions assign each belief a credibility (reflecting source trust) and a confidence (derived from internal structural support). Unlike classical probabilistic models, our approach does not assume prior coherence or require belief updating. Unlike logical and argumentation-based frameworks, it supports fine-grained structural representation without committing to binary justification status or deductive closure. The model is purely static and deliberately excludes inference or revision procedures. Its aim is to provide a foundational substrate for analyzing the internal organization of belief systems, including coherence conditions, epistemic tensions, and representational limits. By distinguishing belief structure from belief strength, this formalism enables a richer classification of epistemic states than existing probabilistic, logical, or argumentation-based approaches. 

**Abstract (ZH)**: 信念体系通常被视为全局一致的命题集合或标量概率分布。这类表示往往会掩盖信念的内部结构，混淆外部可信度与内部一致性，并排除对碎片化或矛盾性认识状态的建模。本文引入了一种针对信念体系的最小形式化方法，即定向加权图。在此框架中，节点表示个体信念，边编码认识关系（如支持或反驳），并通过两个不同的函数为每个信念分配可信度（反映来源信任）和信心（源自内部结构支持）。与经典的概率模型不同，我们的方法不假定先前的一致性或要求信念更新。与基于逻辑和论证的框架不同，它支持精细的结构表示，而无需承诺二元的合理性状态或演绎闭包。该模型完全是静态的，并故意排除推断或修订程序。其目标是为分析信念体系的内部组织提供一个基础性构件，包括一致性条件、认识紧张和表示限制。通过区分信念结构与信念强度，该形式化方法使得对于认识状态的分类比现有的概率、逻辑或基于论证的方法更为丰富。 

---
# Adaptive AI Agent Placement and Migration in Edge Intelligence Systems 

**Title (ZH)**: 边缘智慧行统中自适应AI代理放置与迁移 

**Authors**: Xingdan Wang, Jiayi He, Zhiqing Tang, Jianxiong Guo, Jiong Lou, Liping Qian, Tian Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.03345)  

**Abstract**: The rise of LLMs such as ChatGPT and Claude fuels the need for AI agents capable of real-time task handling. However, migrating data-intensive, multi-modal edge workloads to cloud data centers, traditionally used for agent deployment, introduces significant latency. Deploying AI agents at the edge improves efficiency and reduces latency. However, edge environments present challenges due to limited and heterogeneous resources. Maintaining QoS for mobile users necessitates agent migration, which is complicated by the complexity of AI agents coordinating LLMs, task planning, memory, and external tools. This paper presents the first systematic deployment and management solution for LLM-based AI agents in dynamic edge environments. We propose a novel adaptive framework for AI agent placement and migration in edge intelligence systems. Our approach models resource constraints and latency/cost, leveraging ant colony algorithms and LLM-based optimization for efficient decision-making. It autonomously places agents to optimize resource utilization and QoS and enables lightweight agent migration by transferring only essential state. Implemented on a distributed system using AgentScope and validated across globally distributed edge servers, our solution significantly reduces deployment latency and migration costs. 

**Abstract (ZH)**: LLMs如ChatGPT和Claude的兴起推动了能够实时处理任务的AI代理的需求。然而，将数据密集型、多模态的边缘工作负载迁移到传统用于代理部署的云数据中心，会引入显著的延迟。在边缘部署AI代理可以提高效率并减少延迟。然而，边缘环境由于资源有限且多样化而存在挑战。为了满足移动用户的服务质量要求，需要进行代理迁移，这增加了复杂性，因为AI代理需要协调LLMs、任务规划、内存和外部工具。本文提出了一种系统化的部署和管理解决方案，用于动态边缘环境中基于LLM的AI代理。我们提出了一种新的自适应框架，用于边缘智能系统的AI代理放置和迁移。我们的方法通过利用蚂蚁 colony 算法和基于LLM的优化来建模资源约束和延迟/成本，以实现高效的决策。该框架能够自动放置代理以优化资源利用率和QoS，并通过仅传输必要的状态实现轻量级代理迁移。通过在分布式系统中使用AgentScope并跨全球分布的边缘服务器验证，我们的解决方案显著减少了部署延迟和迁移成本。 

---
# ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools 

**Title (ZH)**: ToolVQA：一种基于外部工具的多步推理VQA数据集 

**Authors**: Shaofeng Yin, Ting Lei, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03284)  

**Abstract**: Integrating external tools into Large Foundation Models (LFMs) has emerged as a promising approach to enhance their problem-solving capabilities. While existing studies have demonstrated strong performance in tool-augmented Visual Question Answering (VQA), recent benchmarks reveal significant gaps in real-world tool-use proficiency, particularly in functionally diverse multimodal settings requiring multi-step reasoning. In this work, we introduce ToolVQA, a large-scale multimodal dataset comprising 23K instances, designed to bridge this gap. Unlike previous datasets that rely on synthetic scenarios and simplified queries, ToolVQA features real-world visual contexts and challenging implicit multi-step reasoning tasks, better aligning with real user interactions. To construct this dataset, we propose ToolEngine, a novel data generation pipeline that employs Depth-First Search (DFS) with a dynamic in-context example matching mechanism to simulate human-like tool-use reasoning. ToolVQA encompasses 10 multimodal tools across 7 diverse task domains, with an average inference length of 2.78 reasoning steps per instance. The fine-tuned 7B LFMs on ToolVQA not only achieve impressive performance on our test set but also surpass the large close-sourced model GPT-3.5-turbo on various out-of-distribution (OOD) datasets, demonstrating strong generalizability to real-world tool-use scenarios. 

**Abstract (ZH)**: 将外部工具整合到大型基础模型中已成为增强其解决问题能力的一种有前途的方法。尽管现有研究在工具增强的视觉问答（VQA）方面展示了强大的性能，但近期基准测试揭示了在需要多步推理的功能多样化多模态设置中，其实用能力存在显著差距。在本工作中，我们引入了ToolVQA，这是一个包含23K实例的大规模多模态数据集，旨在弥合这一差距。与依赖于合成场景和简化查询的先前数据集不同，ToolVQA 包含真实世界的视觉上下文和具有挑战性的隐式多步推理任务，更好地与实际用户交互相契合。为了构建此数据集，我们提出了ToolEngine，这是一种新颖的数据生成流水线，利用深度优先搜索（DFS）与动态上下文示例匹配机制来模拟类似人类的工具使用推理。ToolVQA 包含跨7个不同任务域的10种多模态工具，平均每实例推理步骤数为2.78步。经过ToolVQA 微调的7B 大型基础模型不仅在我们的测试集上取得了令人印象深刻的效果，还在多种离分布（OOD）数据集上超过了闭源大型模型GPT-3.5-turbo，展示了其强大的现实世界工具使用场景泛化能力。 

---
# Full-History Graphs with Edge-Type Decoupled Networks for Temporal Reasoning 

**Title (ZH)**: 带有边类型解耦网络的全历史图Temporal Reasoning 

**Authors**: Osama Mohammed, Jiaxin Pan, Mojtaba Nayyeri, Daniel Hernández, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2508.03251)  

**Abstract**: Modeling evolving interactions among entities is critical in many real-world tasks. For example, predicting driver maneuvers in traffic requires tracking how neighboring vehicles accelerate, brake, and change lanes relative to one another over consecutive frames. Likewise, detecting financial fraud hinges on following the flow of funds through successive transactions as they propagate through the network. Unlike classic time-series forecasting, these settings demand reasoning over who interacts with whom and when, calling for a temporal-graph representation that makes both the relations and their evolution explicit. Existing temporal-graph methods typically use snapshot graphs to encode temporal evolution. We introduce a full-history graph that instantiates one node for every entity at every time step and separates two edge sets: (i) intra-time-step edges that capture relations within a single frame and (ii) inter-time-step edges that connect an entity to itself at consecutive steps. To learn on this graph we design an Edge-Type Decoupled Network (ETDNet) with parallel modules: a graph-attention module aggregates information along intra-time-step edges, a multi-head temporal-attention module attends over an entity's inter-time-step history, and a fusion module combines the two messages after every layer. Evaluated on driver-intention prediction (Waymo) and Bitcoin fraud detection (Elliptic++), ETDNet consistently surpasses strong baselines, lifting Waymo joint accuracy to 75.6\% (vs. 74.1\%) and raising Elliptic++ illicit-class F1 to 88.1\% (vs. 60.4\%). These gains demonstrate the benefit of representing structural and temporal relations as distinct edges in a single graph. 

**Abstract (ZH)**: 建模实体间 evolving 的相互互作用对于解决实际任务至关重要。例如，在预测交通中的驾驶员操作需要跟踪相邻车辆的加速、制动及和车道变化相对于彼此的情况。同样地，，检测金融欺诈依赖于资金在相继交易间如何传播。不同于传统的时间序列预测，这里ET 环境需要推理交互的主体是谁和以及何时，从而需要一个体现关系及其演变的时间图表示表示。现有的时间图方法通常使用快照图来表示时间演变。我们引入了一个多节点历史图，每个实体在每个时间点都有一个独立节点，并并 两个边 时间集：（）单 内时间内的 edges，描述在单一时间内的关系；(ii) � � � � 跨节的 edges，将实体连接到其在连续步骤间的自身。为了构建该图，我们引入了一个边类型解耦网络（ETDNetD �包含两种模块：图注意力模块聚集内时间内的 edges；D � 多头头头暂时注意力模块关注单个实体的跨时间步内的 edgesD；D 以及融合模块在每层将将两种消息合并。在 WayOn（（WayWay） 和 和 � 咯抵币行为检测及 古埃比特币欺诈检测DElliptic++D 中实验结果表明ETET ETDTNet 在在两套实验上均显著优于基线线基线D其中在 WayOnD的联合准确率上从D4...%%提升到D6D..DD在 Elliptic++D 中非法类别准确分离D从D6D.D提升到到D>。这些结果表明了构建单一图中体现结构D暂时关系D使用理解的优点。 

---
# Causal identification with $Y_0$ 

**Title (ZH)**: 因果识别与外生Y因果关系确定 

**Authors**: Charles Tapley Hoyt, Craig Bakker, Richard J. Callahan, Joseph Cottam, August George, Benjamin M. Gyori, Haley M. Hummel, Nathaniel Merrill, Sara Mohammad Taheri, Pruthvi Prakash Navada, Marc-Antoine Parent, Adam Rupe, Olga Vitek, Jeremy Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2508.03167)  

**Abstract**: We present the $Y_0$ Python package, which implements causal identification algorithms that apply interventional, counterfactual, and transportability queries to data from (randomized) controlled trials, observational studies, or mixtures thereof. $Y_0$ focuses on the qualitative investigation of causation, helping researchers determine whether a causal relationship can be estimated from available data before attempting to estimate how strong that relationship is. Furthermore, $Y_0$ provides guidance on how to transform the causal query into a symbolic estimand that can be non-parametrically estimated from the available data. $Y_0$ provides a domain-specific language for representing causal queries and estimands as symbolic probabilistic expressions, tools for representing causal graphical models with unobserved confounders, such as acyclic directed mixed graphs (ADMGs), and implementations of numerous identification algorithms from the recent causal inference literature. The $Y_0$ source code can be found under the MIT License at this https URL and it can be installed with pip install y0. 

**Abstract (ZH)**: 我们介绍$Y_0$ Python包，该包实现了基于介入性、反事实性和可运输性查询的数据因果识别算法，适用于随机对照试验、观察研究或其混合数据。$Y_0$专注于因果关系的定性研究，帮助研究人员在尝试估计关系的强度之前，确定是否可以从可用数据中估计因果关系。此外，$Y_0$提供了将因果查询转化为可以直接从可用数据中非参数化估计的符号估计量的指导。$Y_0$提供了一种领域专用的语言来表示因果查询和估计量作为符号概率表达式，并包含表示包含未观察到混杂因素的因果图形模型的工具，如有向混合无环图（ADMG），以及最近因果推断文献中多种识别算法的实现。$Y_0$的源代码可在MIT License下从该网址获取，并可通过pip install y0安装。 

---
# Can Large Language Models Bridge the Gap in Environmental Knowledge? 

**Title (ZH)**: 大型语言模型能否弥合环境知识鸿沟？ 

**Authors**: Linda Smail, David Santandreu Calonge, Firuz Kamalov, Nur H. Orak  

**Link**: [PDF](https://arxiv.org/pdf/2508.03149)  

**Abstract**: This research investigates the potential of Artificial Intelligence (AI) models to bridge the knowledge gap in environmental education among university students. By focusing on prominent large language models (LLMs) such as GPT-3.5, GPT-4, GPT-4o, Gemini, Claude Sonnet, and Llama 2, the study assesses their effectiveness in conveying environmental concepts and, consequently, facilitating environmental education. The investigation employs a standardized tool, the Environmental Knowledge Test (EKT-19), supplemented by targeted questions, to evaluate the environmental knowledge of university students in comparison to the responses generated by the AI models. The results of this study suggest that while AI models possess a vast, readily accessible, and valid knowledge base with the potential to empower both students and academic staff, a human discipline specialist in environmental sciences may still be necessary to validate the accuracy of the information provided. 

**Abstract (ZH)**: 本研究探讨了人工智能（AI）模型在填补大学学生环境教育知识差距方面的潜力。通过专注于如GPT-3.5、GPT-4、GPT-4o、Gemini、Claude Sonnet和Llama 2等 prominenet大规模语言模型，本研究评估了这些模型在传达环境概念方面的有效性，进而促进环境教育。研究采用了标准化工具环境知识测试（EKT-19），并结合针对性问题，评估了大学学生的环境知识与AI模型生成的回答之间的差异。研究结果表明，虽然AI模型拥有广泛、易于获取且有效的知识库，有可能 empower 学生和学术人员，但仍然需要环境科学领域的专业人士来验证所提供信息的准确性。 

---
# T2UE: Generating Unlearnable Examples from Text Descriptions 

**Title (ZH)**: T2UE: 从文本描述生成不可学会的示例 

**Authors**: Xingjun Ma, Hanxun Huang, Tianwei Song, Ye Sun, Yifeng Gao, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03091)  

**Abstract**: Large-scale pre-training frameworks like CLIP have revolutionized multimodal learning, but their reliance on web-scraped datasets, frequently containing private user data, raises serious concerns about misuse. Unlearnable Examples (UEs) have emerged as a promising countermeasure against unauthorized model training, employing carefully crafted unlearnable noise to disrupt the learning of meaningful representations from protected data. Current approaches typically generate UEs by jointly optimizing unlearnable noise for both images and their associated text descriptions (or labels). However, this optimization process is often computationally prohibitive for on-device execution, forcing reliance on external third-party services. This creates a fundamental privacy paradox: users must initially expose their data to these very services to achieve protection, thereby compromising privacy in the process. Such a contradiction has severely hindered the development of practical, scalable data protection solutions. To resolve this paradox, we introduce \textbf{Text-to-Unlearnable Example (T2UE)}, a novel framework that enables users to generate UEs using only text descriptions. T2UE circumvents the need for original image data by employing a text-to-image (T2I) model to map text descriptions into the image (noise) space, combined with an error-minimization framework to produce effective unlearnable noise. Extensive experiments show that T2UE-protected data substantially degrades performance in downstream tasks (e.g., cross-modal retrieval) for state-of-the-art models. Notably, the protective effect generalizes across diverse architectures and even to supervised learning settings. Our work demonstrates the feasibility of "zero-contact data protection", where personal data can be safeguarded based solely on their textual descriptions, eliminating the need for direct data exposure. 

**Abstract (ZH)**: Large-scale预训练框架如CLIP已经革新了多模态学习，但它们依赖于网页抓取的数据集，经常包含私人用户数据，这引发了关于滥用的严重担忧。不可学习示例（Unlearnable Examples, UEs）作为防止未经授权模型训练的有前景的对策，通过精心构造的不可学习噪声干扰从受保护数据中学习有意义的表示。当前方法通常通过同时优化图像及其关联文本描述（或标签）的不可学习噪声来进行这一过程。然而，这种优化过程在设备上执行时经常是计算上不可行的，迫使依赖于外部第三方服务。这种矛盾创造了一个基础的隐私悖论：用户必须最初向这些服务暴露其数据以实现保护，从而在过程中损害了隐私。这种矛盾严重阻碍了实用、可扩展的数据保护解决方案的发展。为了解决这一悖论，我们引入了**Text-to-Unlearnable Example (T2UE)**，一种全新的框架，使用户能够仅使用文本描述生成UEs。T2UE绕过了对原始图像数据的需求，通过使用文本到图像（Text-to-Image, T2I）模型将文本描述映射到图像（噪声）空间，并结合误差最小化框架生成有效的不可学习噪声。实验结果表明，T2UE保护的数据在下游任务（例如，跨模态检索）中显著降低了最先进的模型的性能。值得注意的是，保护效果在不同的架构中以及监督学习环境中都能泛化。我们的工作展示了“零接触数据保护”的可行性，即基于文本描述即可保护个人数据，而无需直接暴露数据。 

---
# MissDDIM: Deterministic and Efficient Conditional Diffusion for Tabular Data Imputation 

**Title (ZH)**: MissDDIM：确定性和高效条件扩散方法在表格数据插补中的应用 

**Authors**: Youran Zhou, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2508.03083)  

**Abstract**: Diffusion models have recently emerged as powerful tools for missing data imputation by modeling the joint distribution of observed and unobserved variables. However, existing methods, typically based on stochastic denoising diffusion probabilistic models (DDPMs), suffer from high inference latency and variable outputs, limiting their applicability in real-world tabular settings. To address these deficiencies, we present in this paper MissDDIM, a conditional diffusion framework that adapts Denoising Diffusion Implicit Models (DDIM) for tabular imputation. While stochastic sampling enables diverse completions, it also introduces output variability that complicates downstream processing. 

**Abstract (ZH)**: 差分模型最近已成为通过建模可观测和不可观测变量的联合分布来进行缺失数据插补的强大工具。然而，现有方法通常基于随机去噪扩散概率模型（DDPMs），存在推断延迟高和输出可变性高的问题，限制了其在实际表格数据设置中的应用。为解决这些缺陷，本文提出了一个基于去噪扩散隐式模型（DDIM）的条件扩散框架MissDDIM，用于表格数据插补。尽管随机采样能实现多样的完成方式，但也引入了输出可变性，这会使下游处理变得复杂。 

---
# Collab-Solver: Collaborative Solving Policy Learning for Mixed-Integer Linear Programming 

**Title (ZH)**: Collab-Solver: 协作求解策略学习 for 混合整数线性规划 

**Authors**: Siyuan Li, Yifan Yu, Yanchen Deng, Zhihao Zhang, Mengjing Chen, Fangzhou Zhu, Tao Zhong, Jianye Hao, Peng Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2508.03030)  

**Abstract**: Mixed-integer linear programming (MILP) has been a fundamental problem in combinatorial optimization. Previous works have designed a plethora of hard-coded heuristics to accomplish challenging MILP solving with domain knowledge. Driven by the high capability of neural networks, recent research is devoted to replacing manually designed heuristics with learned policies. Although learning-based MILP methods have shown great promise, existing worksindependentlytreatthepolicylearningineachmoduleofMILPsolvers without considering their interdependence, severely hurting the solving speed and quality. To address this issue, we propose a novel multi-agent-based policy learning framework for MILP (Collab-Solver), which can collaboratively optimize the policies for multiple modules. Specifically, we formulate the collaboration of cut selection and branching in MILP solving as a Stackelberg game. Under this formulation, we develop a two-phase learning paradigm to stabilize the collaborative policy learning, where the first phase achieves the data-communicated policy pretraining and the second phase further orchestrates the policy learning for various modules. The jointly learned policy significantly improves the solving performance on both synthetic and large-scale real-world MILP datasets. Moreover, the policies learned by Collab-Solver have also demonstrated excellent generalization abilities across different instance sets. 

**Abstract (ZH)**: 基于多agent的混合整数线性规划策略学习框架（Collab-Solver） 

---
# AQUAH: Automatic Quantification and Unified Agent in Hydrology 

**Title (ZH)**: AQUAH: 自动量化与统一代理在水文学中的应用 

**Authors**: Songkun Yan, Zhi Li, Siyu Zhu, Yixin Wen, Mofan Zhang, Mengye Chen, Jie Cao, Yang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02936)  

**Abstract**: We introduce AQUAH, the first end-to-end language-based agent designed specifically for hydrologic modeling. Starting from a simple natural-language prompt (e.g., 'simulate floods for the Little Bighorn basin from 2020 to 2022'), AQUAH autonomously retrieves the required terrain, forcing, and gauge data; configures a hydrologic model; runs the simulation; and generates a self-contained PDF report. The workflow is driven by vision-enabled large language models, which interpret maps and rasters on the fly and steer key decisions such as outlet selection, parameter initialization, and uncertainty commentary. Initial experiments across a range of U.S. basins show that AQUAH can complete cold-start simulations and produce analyst-ready documentation without manual intervention. The results are judged by hydrologists as clear, transparent, and physically plausible. While further calibration and validation are still needed for operational deployment, these early outcomes highlight the promise of LLM-centered, vision-grounded agents to streamline complex environmental modeling and lower the barrier between Earth observation data, physics-based tools, and decision makers. 

**Abstract (ZH)**: AQUAH：首个基于语言的端到端水文建模代理 

---
# Seemingly Simple Planning Problems are Computationally Challenging: The Countdown Game 

**Title (ZH)**: 看似简单的规划问题其实计算上具有挑战性： Countdown 游戏 

**Authors**: Michael Katz, Harsha Kokel, Sarath Sreedharan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02900)  

**Abstract**: There is a broad consensus that the inability to form long-term plans is one of the key limitations of current foundational models and agents. However, the existing planning benchmarks remain woefully inadequate to truly measure their planning capabilities. Most existing benchmarks either focus on loosely defined tasks like travel planning or end up leveraging existing domains and problems from international planning competitions. While the former tasks are hard to formalize and verify, the latter were specifically designed to test and challenge the weaknesses of existing automated planners. To address these shortcomings, we propose a procedure for creating a planning benchmark centered around the game called Countdown, where a player is expected to form a target number from a list of input numbers through arithmetic operations. We discuss how this problem meets many of the desiderata associated with an ideal benchmark for planning capabilities evaluation. Specifically, the domain allows for an intuitive, natural language description for each problem instance, it is computationally challenging (NP-complete), and the instance space is rich enough that we do not have to worry about memorization. We perform an extensive theoretical analysis, establishing the computational complexity result and demonstrate the advantage of our instance generation procedure over public benchmarks. We evaluate a variety of existing LLM-assisted planning methods on instances generated using our procedure. Our results show that, unlike other domains like 24 Game (a special case of Countdown), our proposed dynamic benchmark remains extremely challenging for existing LLM-based approaches. 

**Abstract (ZH)**: 当前基础模型和代理无法形成长期计划的能力是一个广泛的共识，但现有的规划基准仍远不足以真正测量其规划能力。我们提出了一种基于 Countdown 游戏创建规划基准的程序，要求玩家通过算术运算从给定数字列表中形成目标数字。我们讨论了这个问题如何符合理想规划能力评估基准的诸多标准。具体而言，该领域为每个问题实例提供了直观的自然语言描述，计算上具有挑战性（NP完全问题），并且实例空间足够丰富，无需担心记忆问题。我们进行了广泛理论上分析，建立了计算复杂性结果，并展示了我们实例生成程序相对于现有基准的优势。我们使用我们的程序生成的问题实例评估了多种现有的基于大语言模型的规划方法。我们结果显示，与 24 点游戏等其他领域相比，我们提出的动态基准对于现有的基于大语言模型的方法仍然具有极大的挑战性。 

---
# Recovering Individual-Level Activity Sequences from Location-Based Service Data Using a Novel Transformer-Based Model 

**Title (ZH)**: 基于新型Transformer模型从基于位置服务数据中恢复个体活动序列 

**Authors**: Weiyu Luo, Chenfeng Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02734)  

**Abstract**: Location-Based Service (LBS) data provides critical insights into human mobility, yet its sparsity often yields incomplete trip and activity sequences, making accurate inferences about trips and activities difficult. We raise a research problem: Can we use activity sequences derived from high-quality LBS data to recover incomplete activity sequences at the individual level? This study proposes a new solution, the Variable Selection Network-fused Insertion Transformer (VSNIT), integrating the Insertion Transformer's flexible sequence construction with the Variable Selection Network's dynamic covariate handling capability, to recover missing segments in incomplete activity sequences while preserving existing data. The findings show that VSNIT inserts more diverse, realistic activity patterns, more closely matching real-world variability, and restores disrupted activity transitions more effectively aligning with the target. It also performs significantly better than the baseline model across all metrics. These results highlight VSNIT's superior accuracy and diversity in activity sequence recovery tasks, demonstrating its potential to enhance LBS data utility for mobility analysis. This approach offers a promising framework for future location-based research and applications. 

**Abstract (ZH)**: 基于位置的服务（LBS）数据提供了对人体移动行为的关键洞察，但由于其稀疏性，往往会得到不完整的出行和活动序列，这使得准确推断出行和活动变得困难。我们提出一个研究问题：我们能否利用高质量LBS数据推导出的活动序列来恢复个体层面不完整的活动序列？本研究提出了一种新的解决方案——可变选择网络融合插入变换器（VSNIT），将插入变换器灵活的序列构建能力和可变选择网络动态协变量处理能力相结合，以恢复不完整活动序列中缺失的段落，同时保留现有数据。研究发现，VSNIT插入了更多样化、更现实的活动模式，更接近实际中的变化，并更有效地恢复了被中断的活动过渡，与目标更一致。此外，在所有指标上，VSNIT的表现明显优于基线模型。这些结果突显了VSNIT在活动序列恢复任务中优越的准确性和多样性，展示了其提高LBS数据用于移动性分析效用的潜力。该方法为未来的位置基于研究和应用提供了有前途的框架。 

---
# Planning with Dynamically Changing Domains 

**Title (ZH)**: 动态变化领域中的规划 

**Authors**: Mikhail Soutchanski, Yongmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02697)  

**Abstract**: In classical planning and conformant planning, it is assumed that there are finitely many named objects given in advance, and only they can participate in actions and in fluents. This is the Domain Closure Assumption (DCA). However, there are practical planning problems where the set of objects changes dynamically as actions are performed; e.g., new objects can be created, old objects can be destroyed. We formulate the planning problem in first-order logic, assume an initial theory is a finite consistent set of fluent literals, discuss when this guarantees that in every situation there are only finitely many possible actions, impose a finite integer bound on the length of the plan, and propose to organize search over sequences of actions that are grounded at planning time. We show the soundness and completeness of our approach. It can be used to solve the bounded planning problems without DCA that belong to the intersection of sequential generalized planning (without sensing actions) and conformant planning, restricted to the case without the disjunction over fluent literals. We discuss a proof-of-the-concept implementation of our planner. 

**Abstract (ZH)**: 在经典规划和一致规划中，假设存在一些预先命名的对象，并且只有这些对象可以参与动作和状态变化，这被称为域闭包假设（DCA）。然而，有些实际规划问题中，随着动作的执行，对象的集合会动态变化；例如，可以创建新对象，也可以销毁旧对象。我们采用一阶逻辑来形式化规划问题，假设初始理论是一组有限的一致的质蜀公式集，讨论在这种情况下保证每个情况下仅有有限数量的动作，规定计划的长度上限为有限整数，并提出在规划时基于动作序列组织搜索。我们证明了该方法的正确性和完备性。该方法可用于解决符合DCA的受限于顺序泛化规划（不包含感知动作）和一致规划交集的有界规划问题，且不涉及质蜀公式的析取。我们讨论了我们规划器的概念性实现。 

---
# Classifying Epistemic Relationships in Human-AI Interaction: An Exploratory Approach 

**Title (ZH)**: 人类-人工智能互动中的知识关系分类：一种探索性方法 

**Authors**: Shengnan Yang, Rongqian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.03673)  

**Abstract**: As AI systems become integral to knowledge-intensive work, questions arise not only about their functionality but also their epistemic roles in human-AI interaction. While HCI research has proposed various AI role typologies, it often overlooks how AI reshapes users' roles as knowledge contributors. This study examines how users form epistemic relationships with AI-how they assess, trust, and collaborate with it in research and teaching contexts. Based on 31 interviews with academics across disciplines, we developed a five-part codebook and identified five relationship types: Instrumental Reliance, Contingent Delegation, Co-agency Collaboration, Authority Displacement, and Epistemic Abstention. These reflect variations in trust, assessment modes, tasks, and human epistemic status. Our findings show that epistemic roles are dynamic and context-dependent. We argue for shifting beyond static metaphors of AI toward a more nuanced framework that captures how humans and AI co-construct knowledge, enriching HCI's understanding of the relational and normative dimensions of AI use. 

**Abstract (ZH)**: 随着人工智能系统成为知识密集型工作的一部分，不仅对其功能产生了疑问，还对其在人机交互中的认知角色产生了疑问。虽然HCI研究提出了多种人工智能角色类型，但往往忽略了人工智能如何重塑用户作为知识贡献者的角色。本研究考察了用户在研究和教学情境中与人工智能形成认知关系的方式——他们如何评估、信任并与其合作。基于与跨学科21位学者的访谈，我们制定了一个五部分的编码手册并识别出五种关系类型：工具依赖、条件委托、共主体制作、权威替代和认知回避。这些反映了信任程度、评估模式、任务及人类认知地位的变化。研究发现表明，认知角色是动态的且依赖于情境。我们主张超越静态的人工智能元喻，转向一个更细致的框架，以捕捉人类和人工智能如何共同构建知识的过程，丰富HCI对人工智能使用关系和规范维度的理解。 

---
# Beyond risk: A proto-framework for assessing the societal impact of AI systems 

**Title (ZH)**: 超越风险：评估人工智能系统社会影响的雏形框架 

**Authors**: Willem Fourie  

**Link**: [PDF](https://arxiv.org/pdf/2508.03666)  

**Abstract**: In the discourse on AI regulation, 'responsible AI' is the dominant paradigm, with the focus on mitigating the risks related to AI systems. While this focus is important and necessary, it has limited use for a systematic consideration of AI's societal impact. This paper proposes a proto-framework for assessing the societal impact of AI systems by operationalising the concept of freedom. This proto-framework is intended as a step towards a fully operationalised framework to be used in policymaking contexts. By drawing on Kantian philosophy and related contemporary interpretations, freedom is developed as the counterpart to the concept of responsibility. Two dimensions of freedom are developed in further detail: freedom as capability and freedom as opportunity. These two dimensions of freedom are then applied in a proto-framework that systematically considers AI's impact on society using the Sustainable Development Goals. This proto-framework aims to complement current risk-based approaches and thereby offers a first step towards operationalising the concept of freedom in AI regulation. 

**Abstract (ZH)**: 人工智能系统的社会影响评估框架：自由概念的操作化 

---
# Forest vs Tree: The $(N, K)$ Trade-off in Reproducible ML Evaluation 

**Title (ZH)**: 森林与树木：可重现ML评估中的$(N, K)$权衡 

**Authors**: Deepak Pandita, Flip Korn, Chris Welty, Christopher M. Homan  

**Link**: [PDF](https://arxiv.org/pdf/2508.03663)  

**Abstract**: Reproducibility is a cornerstone of scientific validation and of the authority it confers on its results. Reproducibility in machine learning evaluations leads to greater trust, confidence, and value. However, the ground truth responses used in machine learning often necessarily come from humans, among whom disagreement is prevalent, and surprisingly little research has studied the impact of effectively ignoring disagreement in these responses, as is typically the case. One reason for the lack of research is that budgets for collecting human-annotated evaluation data are limited, and obtaining more samples from multiple annotators for each example greatly increases the per-item annotation costs. We investigate the trade-off between the number of items ($N$) and the number of responses per item ($K$) needed for reliable machine learning evaluation. We analyze a diverse collection of categorical datasets for which multiple annotations per item exist, and simulated distributions fit to these datasets, to determine the optimal $(N, K)$ configuration, given a fixed budget ($N \times K$), for collecting evaluation data and reliably comparing the performance of machine learning models. Our findings show, first, that accounting for human disagreement may come with $N \times K$ at no more than 1000 (and often much lower) for every dataset tested on at least one metric. Moreover, this minimal $N \times K$ almost always occurred for $K > 10$. Furthermore, the nature of the tradeoff between $K$ and $N$ -- or if one even existed -- depends on the evaluation metric, with metrics that are more sensitive to the full distribution of responses performing better at higher levels of $K$. Our methods can be used to help ML practitioners get more effective test data by finding the optimal metrics and number of items and annotations per item to collect to get the most reliability for their budget. 

**Abstract (ZH)**: 机器学习评估中的人类分歧及其影响的 reproducibility 对科学验证及其成果权威性的基石作用。机器学习评估中的再现性可增强信任、信心和价值。然而，机器学习中使用的 ground truth 常常来源于人类，而人类之间存在分歧，令人惊讶的是，很少有研究探讨忽略这些分歧的影响，这通常是在常规操作中发生的。造成研究不足的一个原因是收集人类标注评价数据的预算有限，从每个示例获取多个标注者的更多样本会大大增加每项标注成本。我们研究了可靠机器学习评估所需项目数量($N$)与每个项目所需响应数量($K$)之间的权衡。我们分析了多个标注存在的多样化的分类数据集和适用于这些数据集的模拟分布，以确定给定固定预算($N \times K$)时，收集评价数据并可靠比较机器学习模型性能的最佳($N, K$)配置。我们的发现表明，首先，考虑人类分歧可能只需要不超过1000（通常更低）的$N \times K$，这适用于至少在一个度量标准上测试的所有数据集。此外，这一最小的$N \times K$几乎总是发生在$K > 10$的情况下。进一步而言，$K$与$N$之间的权衡性质——或者是否存在权衡——取决于评估指标，更加敏感于完整响应分布的指标在较高$K$水平上表现更好。我们的方法可以帮助机器学习 practitioner 获取更有效的测试数据，通过找到最佳的度量标准和收集项目的数量及标注数量来最大化预算的可靠性。 

---
# Probing the Gaps in ChatGPT Live Video Chat for Real-World Assistance for People who are Blind or Visually Impaired 

**Title (ZH)**: 探究ChatGPT实时视频聊天中的缺口以实现在盲人或视觉障碍者中提供实际帮助的可能性 

**Authors**: Ruei-Che Chang, Rosiana Natalie, Wenqian Xu, Jovan Zheng Feng Yap, Anhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.03651)  

**Abstract**: Recent advancements in large multimodal models have provided blind or visually impaired (BVI) individuals with new capabilities to interpret and engage with the real world through interactive systems that utilize live video feeds. However, the potential benefits and challenges of such capabilities to support diverse real-world assistive tasks remain unclear. In this paper, we present findings from an exploratory study with eight BVI participants. Participants used ChatGPT's Advanced Voice with Video, a state-of-the-art live video AI released in late 2024, in various real-world scenarios, from locating objects to recognizing visual landmarks, across unfamiliar indoor and outdoor environments. Our findings indicate that current live video AI effectively provides guidance and answers for static visual scenes but falls short in delivering essential live descriptions required in dynamic situations. Despite inaccuracies in spatial and distance information, participants leveraged the provided visual information to supplement their mobility strategies. Although the system was perceived as human-like due to high-quality voice interactions, assumptions about users' visual abilities, hallucinations, generic responses, and a tendency towards sycophancy led to confusion, distrust, and potential risks for BVI users. Based on the results, we discuss implications for assistive video AI agents, including incorporating additional sensing capabilities for real-world use, determining appropriate intervention timing beyond turn-taking interactions, and addressing ecological and safety concerns. 

**Abstract (ZH)**: 最近在大型多模态模型方面的进展为盲人或视力受损（BVI）个体提供了新的能力，通过利用实时视频流的交互系统来解读和参与现实世界。然而，这些能力支持多样化现实世界辅助任务的潜在优势和挑战尚不明确。在本文中，我们报告了与八名BVI参与者进行的探索性研究结果。参与者在各种现实世界场景中使用了_chatGPT的高级语音加视频功能，这是一种于2024年底发布的先进实时视频AI，在室内和室外环境中从定位物体到识别视觉地标。我们的研究发现，当前的实时视频AI在提供静态视觉场景指导和答案方面效果显著，但在动态情况下的实时描述方面存在不足。尽管存在关于空间和距离信息的不准确性，参与者仍利用提供的视觉信息来补充他们的移动策略。尽管系统由于高质量的语音交互被认为具有人性化，但关于用户视觉能力的假设、幻觉、通用回应以及奉承倾向导致了BVI用户的困惑、不信任和潜在风险。基于研究结果，我们讨论了辅助视频AI代理的潜在影响，包括在现实世界使用中增加额外的传感能力、确定适当的介入时机以及解决生态学和安全问题。 

---
# Cross-Model Semantics in Representation Learning 

**Title (ZH)**: 跨模型语义在表示学习中的研究 

**Authors**: Saleh Nikooroo, Thomas Engel  

**Link**: [PDF](https://arxiv.org/pdf/2508.03649)  

**Abstract**: The internal representations learned by deep networks are often sensitive to architecture-specific choices, raising questions about the stability, alignment, and transferability of learned structure across models. In this paper, we investigate how structural constraints--such as linear shaping operators and corrective paths--affect the compatibility of internal representations across different architectures. Building on the insights from prior studies on structured transformations and convergence, we develop a framework for measuring and analyzing representational alignment across networks with distinct but related architectural priors. Through a combination of theoretical insights, empirical probes, and controlled transfer experiments, we demonstrate that structural regularities induce representational geometry that is more stable under architectural variation. This suggests that certain forms of inductive bias not only support generalization within a model, but also improve the interoperability of learned features across models. We conclude with a discussion on the implications of representational transferability for model distillation, modular learning, and the principled design of robust learning systems. 

**Abstract (ZH)**: 深度网络学习到的内部表示对架构特定选择的高度敏感，引发了关于跨模型学习结构的稳定性和转移性的疑问。本文探讨了结构性约束（如线性塑造算子和矫正路径）如何影响不同架构下内部表示的兼容性。基于先前关于结构转换和收敛研究中的见解，我们构建了一个框架，用于衡量和分析具有不同但相关架构先验的网络之间的表示对齐程度。通过理论洞察、实证探究和受控的迁移实验，我们展示出结构性规律诱导出的表示几何在架构变化下更加稳定。这表明某些形式的归纳偏置不仅支持模型内部的泛化，还能提高跨模型学习特征的互操作性。最后，我们讨论了表示迁移性对模型精简、模块化学习以及稳健学习系统设计的意义。 

---
# Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction 

**Title (ZH)**: Goedel-Prover-V2: 通过支架式数据合成和自我修正扩展形式定理证明 

**Authors**: Yong Lin, Shange Tang, Bohan Lyu, Ziran Yang, Jui-Hui Chung, Haoyu Zhao, Lai Jiang, Yihan Geng, Jiawei Ge, Jingruo Sun, Jiayun Wu, Jiri Gesi, Ximing Lu, David Acuna, Kaiyu Yang, Hongzhou Lin, Yejin Choi, Danqi Chen, Sanjeev Arora, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03613)  

**Abstract**: We introduce Goedel-Prover-V2, a series of open-source language models that set a new state-of-the-art in automated theorem proving. Built on the standard expert iteration and reinforcement learning pipeline, our approach incorporates three key innovations: (1) Scaffolded data synthesis: We generate synthetic tasks of increasing difficulty to train the model to master increasingly complex theorems; (2) Verifier-guided self-correction: We enable the model to iteratively revise its proofs by leveraging feedback from the Lean compiler; (3) Model averaging: We merge model checkpoints to mitigate the decrease in model output diversity in later stages of training. Our small model, Goedel-Prover-V2-8B, reaches 84.6% pass@32 on MiniF2F and outperforms DeepSeek-Prover-V2-671B under the same metric, despite being 80X smaller. Our flagship model, Goedel-Prover-V2-32B, achieves 88.1% on MiniF2F at pass@32 in standard mode and 90.4% in self-correction mode, outperforming prior SOTA by a large margin. Additionally, our flagship model solves 86 problems on PutnamBench at pass@184, securing the first place among open-source models on the leaderboard, surpassing DeepSeek-Prover-V2-671B's record of solving 47 problems by pass@1024 with a significantly smaller model size and compute budget. At the time of its release (July-August 2025), Goedel-Prover-V2 achieves the strongest overall performance among all open-source theorem provers. It also ranks among the top-performing models--including closed-source systems with publicly reported performance--under a constrained test-time compute budget. Our models, code, and data are released at this https URL. 

**Abstract (ZH)**: Goedel-Prover-V2：一种开源语言模型系列，在自动定理证明中达到新的人工智能最佳水平 

---
# DeepFaith: A Domain-Free and Model-Agnostic Unified Framework for Highly Faithful Explanations 

**Title (ZH)**: DeepFaith: 一种无需领域知识且模型无关的统一解释框架，实现高度忠实的解释 

**Authors**: Yuhan Guo, Lizhong Ding, Shihan Jia, Yanyu Ren, Pengqi Li, Jiarun Fu, Changsheng Li, Ye yuan, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03586)  

**Abstract**: Explainable AI (XAI) builds trust in complex systems through model attribution methods that reveal the decision rationale. However, due to the absence of a unified optimal explanation, existing XAI methods lack a ground truth for objective evaluation and optimization. To address this issue, we propose Deep architecture-based Faith explainer (DeepFaith), a domain-free and model-agnostic unified explanation framework under the lens of faithfulness. By establishing a unified formulation for multiple widely used and well-validated faithfulness metrics, we derive an optimal explanation objective whose solution simultaneously achieves optimal faithfulness across these metrics, thereby providing a ground truth from a theoretical perspective. We design an explainer learning framework that leverages multiple existing explanation methods, applies deduplicating and filtering to construct high-quality supervised explanation signals, and optimizes both pattern consistency loss and local correlation to train a faithful explainer. Once trained, DeepFaith can generate highly faithful explanations through a single forward pass without accessing the model being explained. On 12 diverse explanation tasks spanning 6 models and 6 datasets, DeepFaith achieves the highest overall faithfulness across 10 metrics compared to all baseline methods, highlighting its effectiveness and cross-domain generalizability. 

**Abstract (ZH)**: 基于深度架构的信实解释器（DeepFaith）：一种无领域依赖且模型无关的统一解释框架 

---
# Decoding and Engineering the Phytobiome Communication for Smart Agriculture 

**Title (ZH)**: 解码和工程化植物组微生物通信以实现智能农业 

**Authors**: Fatih Gulec, Hamdan Awan, Nigel Wallbridge, Andrew W. Eckford  

**Link**: [PDF](https://arxiv.org/pdf/2508.03584)  

**Abstract**: Smart agriculture applications, integrating technologies like the Internet of Things and machine learning/artificial intelligence (ML/AI) into agriculture, hold promise to address modern challenges of rising food demand, environmental pollution, and water scarcity. Alongside the concept of the phytobiome, which defines the area including the plant, its environment, and associated organisms, and the recent emergence of molecular communication (MC), there exists an important opportunity to advance agricultural science and practice using communication theory. In this article, we motivate to use the communication engineering perspective for developing a holistic understanding of the phytobiome communication and bridge the gap between the phytobiome communication and smart agriculture. Firstly, an overview of phytobiome communication via molecular and electrophysiological signals is presented and a multi-scale framework modeling the phytobiome as a communication network is conceptualized. Then, how this framework is used to model electrophysiological signals is demonstrated with plant experiments. Furthermore, possible smart agriculture applications, such as smart irrigation and targeted delivery of agrochemicals, through engineering the phytobiome communication are proposed. These applications merge ML/AI methods with the Internet of Bio-Nano-Things enabled by MC and pave the way towards more efficient, sustainable, and eco-friendly agricultural production. Finally, the implementation challenges, open research issues, and industrial outlook for these applications are discussed. 

**Abstract (ZH)**: 基于物联网和机器学习/人工智能技术的智能农业应用通过整合通信理论有望解决现代粮食需求上升、环境污染和水资源短缺等挑战。结合植微组的概念以及分子通信的新兴领域，利用通信工程视角理解植微组通信并构建植微组通信与智能农业之间的桥梁具有重要意义。本文通过分子和电生理信号概述植微组通信，构建多层次框架将植微组视为通信网络，并通过植物实验展示了该框架的应用。此外，提出通过工程化植微组通信实现智能灌溉和精准农业化学品输送等智能农业应用，这些应用融合了基于分子通信的生物纳米物联网与机器学习/人工智能方法，促进了更为高效、可持续和环保的农业生产。最后，讨论了这些应用的实施挑战、开放研究问题及产业前景。 

---
# Supervised Dynamic Dimension Reduction with Deep Neural Network 

**Title (ZH)**: 监督动态维度缩减与深度神经网络 

**Authors**: Zhanye Luo, Yuefeng Han, Xiufan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03546)  

**Abstract**: This paper studies the problem of dimension reduction, tailored to improving time series forecasting with high-dimensional predictors. We propose a novel Supervised Deep Dynamic Principal component analysis (SDDP) framework that incorporates the target variable and lagged observations into the factor extraction process. Assisted by a temporal neural network, we construct target-aware predictors by scaling the original predictors in a supervised manner, with larger weights assigned to predictors with stronger forecasting power. A principal component analysis is then performed on the target-aware predictors to extract the estimated SDDP factors. This supervised factor extraction not only improves predictive accuracy in the downstream forecasting task but also yields more interpretable and target-specific latent factors. Building upon SDDP, we propose a factor-augmented nonlinear dynamic forecasting model that unifies a broad family of factor-model-based forecasting approaches. To further demonstrate the broader applicability of SDDP, we extend our studies to a more challenging scenario when the predictors are only partially observable. We validate the empirical performance of the proposed method on several real-world public datasets. The results show that our algorithm achieves notable improvements in forecasting accuracy compared to state-of-the-art methods. 

**Abstract (ZH)**: 本文研究了高维预测因子下时间序列预测的降维问题，提出了一种新的监督深度动态主成分分析（SDDP）框架，将目标变量和滞后观察值纳入因子提取过程。借助时间神经网络，我们通过监督方式按比例调整原始预测因子，并赋予具有更强预测能力的预测因子更大的权重，构建目标感知预测因子。随后对目标感知预测因子进行主成分分析，提取估计的SDDP因子。这种监督因子提取不仅提高了下游预测任务的预测准确性，还得到了更具可解释性和目标特异性的潜在因子。在SDDP的基础上，我们提出了一种因子增强的非线性动态预测模型，统一了基于因子模型的各种预测方法。为进一步展示SDDP的更广泛适用性，我们将其研究扩展到了预测因子部分可观测的更具挑战性的场景。我们在多个真实世界的公开数据集上验证了所提出方法的实证性能。结果显示，与现有最先进的方法相比，我们的算法在预测准确性方面取得了显著改进。 

---
# Retinal Lipidomics Associations as Candidate Biomarkers for Cardiovascular Health 

**Title (ZH)**: 视网膜脂质组学关联作为心血管健康候选生物标志物 

**Authors**: Inamullah, Imran Razzak, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2508.03538)  

**Abstract**: Retinal microvascular imaging is increasingly recognised as a non invasive method for evaluating systemic vascular and metabolic health. However, the association between lipidomics and retinal vasculature remains inadequate. This study investigates the relationships between serum lipid subclasses, free fatty acids (FA), diacylglycerols (DAG), triacylglycerols (TAG), and cholesteryl esters (CE), and retinal microvascular characteristics in a large population-based cohort. Using Spearman correlation analysis, we examined the interconnection between lipid subclasses and ten retinal microvascular traits, applying the Benjamini-Hochberg false discovery rate (BH-FDR) to adjust for statistical significance.
Results indicated that FA were linked to retinal vessel twistiness, while CE correlated with the average widths of arteries and veins. Conversely, DAG and TAG showed negative correlations with the width and complexity of arterioles and venules. These findings suggest that retinal vascular architecture reflects distinct circulating lipid profiles, supporting its role as a non-invasive marker of systemic metabolic health. This study is the first to integrate deep learning (DL)derived retinal traits with lipidomic subclasses in a healthy cohort, thereby providing insights into microvascular structural changes independent of disease status or treatment effects. 

**Abstract (ZH)**: 视网膜微血管成像作为一种无创方法，越来越多地被认可为评估全身血管和代谢健康的有效手段。然而，脂质组学与视网膜微血管之间关联仍不充分。本研究调查了血清脂质亚类、游离脂肪酸（FA）、二酰甘油（DAG）、三酰甘油（TAG）和胆固醇酯（CE）与大规模人群队列中视网膜微血管特征之间的关系。通过Spearman相关分析，研究了脂质亚类与十种视网膜微血管特征之间的联系，并应用Benjamini-Hochberg假发现率（BH-FDR）校正统计显著性。结果显示，游离脂肪酸与视网膜血管的曲折性相关，而胆固醇酯与动脉和静脉的平均宽度相关。相比之下，二酰甘油和三酰甘油与动脉和静脉毛细血管的宽度和复杂性呈负相关。这些发现表明，视网膜血管结构反映出不同的血液循环脂质谱型，支持其作为全身代谢健康非侵入性标志物的作用。本研究是首次将深度学习（DL）提取的视网膜特征与脂质组学亚类整合在健康队列中，从而提供了独立于疾病状态或治疗效应的微血管结构变化的见解。 

---
# CF-RAG: A Dataset and Method for Carbon Footprint QA Using Retrieval-Augmented Generation 

**Title (ZH)**: CF-RAG：一种基于检索增强生成的碳足迹问答数据集与方法 

**Authors**: Kaiwen Zhao, Bharathan Balaji, Stephen Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.03489)  

**Abstract**: Product sustainability reports provide valuable insights into the environmental impacts of a product and are often distributed in PDF format. These reports often include a combination of tables and text, which complicates their analysis. The lack of standardization and the variability in reporting formats further exacerbate the difficulty of extracting and interpreting relevant information from large volumes of documents. In this paper, we tackle the challenge of answering questions related to carbon footprints within sustainability reports available in PDF format. Unlike previous approaches, our focus is on addressing the difficulties posed by the unstructured and inconsistent nature of text extracted from PDF parsing. To facilitate this analysis, we introduce CarbonPDF-QA, an open-source dataset containing question-answer pairs for 1735 product report documents, along with human-annotated answers. Our analysis shows that GPT-4o struggles to answer questions with data inconsistencies. To address this limitation, we propose CarbonPDF, an LLM-based technique specifically designed to answer carbon footprint questions on such datasets. We develop CarbonPDF by fine-tuning Llama 3 with our training data. Our results show that our technique outperforms current state-of-the-art techniques, including question-answering (QA) systems finetuned on table and text data. 

**Abstract (ZH)**: 产品可持续报告提供了有关产品环境影响的宝贵见解，通常以PDF格式分发。这些报告通常包含表格和文本的组合，这使得它们的分析变得复杂。由于缺乏标准化和报告格式的差异性，从大量文档中提取和解释相关信息变得更加困难。本文旨在应对问答系统在处理以PDF格式提供的可持续报告中的碳足迹相关问题时所面临的挑战。与以往的方法不同，我们的重点在于解决从PDF解析中提取的非结构化和不一致文本所带来的难题。为了便于这种分析，我们引入了CarbonPDF-QA，这是一个包含1735个产品报告文档的问题-答案对的数据集，其中包含人工注释的答案。我们的分析表明，GPT-4o在处理数据不一致的问题上表现不佳。为了克服这一局限，我们提出了CarbonPDF，这是一种基于大语言模型的技术，专门针对此类数据集回答碳足迹问题。我们通过使用训练数据对Llama 3模型进行微调来开发CarbonPDF。我们的结果显示，我们的技术在当前最先进的技术中表现最佳，包括那些在表格和文本数据上进行微调的问题回答系统。 

---
# When Cars Have Stereotypes: Auditing Demographic Bias in Objects from Text-to-Image Models 

**Title (ZH)**: 当汽车也有刻板印象：文本到图像模型中对象的 demographic 偏差审核 

**Authors**: Dasol Choi Jihwan Lee, Minjae Lee, Minsuk Kahng  

**Link**: [PDF](https://arxiv.org/pdf/2508.03483)  

**Abstract**: While prior research on text-to-image generation has predominantly focused on biases in human depictions, we investigate a more subtle yet pervasive phenomenon: demographic bias in generated objects (e.g., cars). We introduce SODA (Stereotyped Object Diagnostic Audit), a novel framework for systematically measuring such biases. Our approach compares visual attributes of objects generated with demographic cues (e.g., "for young people'') to those from neutral prompts, across 2,700 images produced by three state-of-the-art models (GPT Image-1, Imagen 4, and Stable Diffusion) in five object categories. Through a comprehensive analysis, we uncover strong associations between specific demographic groups and visual attributes, such as recurring color patterns prompted by gender or ethnicity cues. These patterns reflect and reinforce not only well-known stereotypes but also more subtle and unintuitive biases. We also observe that some models generate less diverse outputs, which in turn amplifies the visual disparities compared to neutral prompts. Our proposed auditing framework offers a practical approach for testing, revealing how stereotypes still remain embedded in today's generative models. We see this as an essential step toward more systematic and responsible AI development. 

**Abstract (ZH)**: 文本到图像生成中的深层群体偏差诊断审计：SODA框架 

---
# fact check AI at SemEval-2025 Task 7: Multilingual and Crosslingual Fact-checked Claim Retrieval 

**Title (ZH)**: SemEval-2025 任务7: 多语言与跨语言事实核查声明检索中的人工智能事实核查 

**Authors**: Pranshu Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2508.03475)  

**Abstract**: SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval is approached as a Learning-to-Rank task using a bi-encoder model fine-tuned from a pre-trained transformer optimized for sentence similarity. Training used both the source languages and their English translations for multilingual retrieval and only English translations for cross-lingual retrieval. Using lightweight models with fewer than 500M parameters and training on Kaggle T4 GPUs, the method achieved 92% Success@10 in multilingual and 80% Success@10 in 5th in crosslingual and 10th in multilingual tracks. 

**Abstract (ZH)**: SemEval-2025 任务7：多语言和跨语言查证断言检索被 Treat 为一种基于双编码器模型的排序学习任务，该模型从一个针对句间相似性优化的预训练变换器微调而来。训练时使用了源语言及其英文翻译进行多语言检索，仅使用英文翻译进行跨语言检索。使用轻量级模型（参数少于500M）并在Kaggle T4 GPU上训练，该方法在多语言赛道上取得了92%的成功率@10，在跨语言赛道上排名第五，多语言赛道上排名第十。 

---
# SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering 

**Title (ZH)**: SonicMaster: 向可控一体化音乐修复与母带处理迈进 

**Authors**: Jan Melechovsky, Ambuj Mehrish, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2508.03448)  

**Abstract**: Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration. To train this model, we construct the SonicMaster dataset, a large dataset of paired degraded and high-quality tracks by simulating common degradation types with nineteen degradation functions belonging to five enhancements groups: equalization, dynamics, reverb, amplitude, and stereo. Our approach leverages a flow-matching generative training paradigm to learn an audio transformation that maps degraded inputs to their cleaned, mastered versions guided by text prompts. Objective audio quality metrics demonstrate that SonicMaster significantly improves sound quality across all artifact categories. Furthermore, subjective listening tests confirm that listeners prefer SonicMaster's enhanced outputs over the original degraded audio, highlighting the effectiveness of our unified approach. 

**Abstract (ZH)**: 音乐录音常常受到诸如过度混响、失真、削波、音调失衡和立体声图像狭窄等音频质量问题的影响，尤其是在缺乏专业设备和 expertise 的非专业环境中制作时更为明显。这些问题通常需要使用专门的工具和手动调整来修正。本文介绍了 SonicMaster，这是一个首个统一的生成模型，用于音乐恢复和母带处理，能够通过基于文本的控制解决广泛种类的 audio artifactual。SonicMaster 可根据自然语言指令应用目标增强，或者以自动模式进行一般性恢复。为了训练此模型，我们构建了 SonicMaster 数据集，这是一个包含大量降质和高质量音频对的大型数据集，通过 19 种代表五类增强效果（均衡、动态、混响、幅度和立体声）的降质函数模拟常见降质类型。我们的方法采用了流匹配生成训练范式，通过文本提示引导学习音频转换，将降质输入映射到其清洁和母带处理版本。客观的音频质量指标表明，SonicMaster 显著提高了所有类别 audio artifact 的音质。此外，主观听觉测试证实听众更偏好 SonicMaster 增强后的输出，突显了我们统一方法的有效性。 

---
# Spatial Imputation Drives Cross-Domain Alignment for EEG Classification 

**Title (ZH)**: 基于空间插值的跨域对齐方法用于EEG分类 

**Authors**: Hongjun Liu, Chao Yao, Yalan Zhang, Xiaokun wang, Xiaojuan Ban  

**Link**: [PDF](https://arxiv.org/pdf/2508.03437)  

**Abstract**: Electroencephalogram (EEG) signal classification faces significant challenges due to data distribution shifts caused by heterogeneous electrode configurations, acquisition protocols, and hardware discrepancies across domains. This paper introduces IMAC, a novel channel-dependent mask and imputation self-supervised framework that formulates the alignment of cross-domain EEG data shifts as a spatial time series imputation task. To address heterogeneous electrode configurations in cross-domain scenarios, IMAC first standardizes different electrode layouts using a 3D-to-2D positional unification mapping strategy, establishing unified spatial representations. Unlike previous mask-based self-supervised representation learning methods, IMAC introduces spatio-temporal signal alignment. This involves constructing a channel-dependent mask and reconstruction task framed as a low-to-high resolution EEG spatial imputation problem. Consequently, this approach simulates cross-domain variations such as channel omissions and temporal instabilities, thus enabling the model to leverage the proposed imputer for robust signal alignment during inference. Furthermore, IMAC incorporates a disentangled structure that separately models the temporal and spatial information of the EEG signals separately, reducing computational complexity while enhancing flexibility and adaptability. Comprehensive evaluations across 10 publicly available EEG datasets demonstrate IMAC's superior performance, achieving state-of-the-art classification accuracy in both cross-subject and cross-center validation scenarios. Notably, IMAC shows strong robustness under both simulated and real-world distribution shifts, surpassing baseline methods by up to $35$\% in integrity scores while maintaining consistent classification accuracy. 

**Abstract (ZH)**: 基于自监督框架的跨域脑电信号分类方法：IMAC 

---
# The Science Fiction Science Method 

**Title (ZH)**: 科幻中的科学方法 

**Authors**: Iyad Rahwan, Azim Shariff, Jean-François Bonnefon  

**Link**: [PDF](https://arxiv.org/pdf/2508.03430)  

**Abstract**: Predicting the social and behavioral impact of future technologies, before they are achieved, would allow us to guide their development and regulation before these impacts get entrenched. Traditionally, this prediction has relied on qualitative, narrative methods. Here we describe a method which uses experimental methods to simulate future technologies, and collect quantitative measures of the attitudes and behaviors of participants assigned to controlled variations of the future. We call this method 'science fiction science'. We suggest that the reason why this method has not been fully embraced yet, despite its potential benefits, is that experimental scientists may be reluctant to engage in work facing such serious validity threats as science fiction science. To address these threats, we consider possible constraints on the kind of technology that science fiction science may study, as well as the unconventional, immersive methods that science fiction science may require. We seek to provide perspective on the reasons why this method has been marginalized for so long, what benefits it would bring if it could be built on strong yet unusual methods, and how we can normalize these methods to help the diverse community of science fiction scientists to engage in a virtuous cycle of validity improvement. 

**Abstract (ZH)**: 预测未来技术的社会和行为影响，以在这些技术实现之前指导其发展和监管，从而防止这些影响固化。传统的预测方法依赖于定性的叙述性方法。在这里，我们描述了一种使用实验方法模拟未来技术并收集参与者在受控版本中的态度和行为的量化指标的方法。我们称之为“科幻科学”。我们建议，尽管这种方法具有潜在益处，但未被完全接受的原因可能是实验科学家可能不愿参与可能会面临严重有效性和现实性威胁的科幻科学工作。为应对这些威胁，我们考虑了科幻科学可能研究的技术类型限制，以及科幻科学可能需要的非传统沉浸式方法。我们旨在提供关于为什么这种方法长期被边缘化的视角，如果能够建立在强而独特的研究方法之上，它将带来哪些好处，以及如何使这些方法正常化以帮助科幻科学家社区建立一个有效的验证改进循环。 

---
# SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models 

**Title (ZH)**: SCFlow: 通过流模型隐式学习风格和内容分离 

**Authors**: Pingchuan Ma, Xiaopei Yang, Yusong Li, Ming Gui, Felix Krause, Johannes Schusterbauer, Björn Ommer  

**Link**: [PDF](https://arxiv.org/pdf/2508.03402)  

**Abstract**: Explicitly disentangling style and content in vision models remains challenging due to their semantic overlap and the subjectivity of human perception. Existing methods propose separation through generative or discriminative objectives, but they still face the inherent ambiguity of disentangling intertwined concepts. Instead, we ask: Can we bypass explicit disentanglement by learning to merge style and content invertibly, allowing separation to emerge naturally? We propose SCFlow, a flow-matching framework that learns bidirectional mappings between entangled and disentangled representations. Our approach is built upon three key insights: 1) Training solely to merge style and content, a well-defined task, enables invertible disentanglement without explicit supervision; 2) flow matching bridges on arbitrary distributions, avoiding the restrictive Gaussian priors of diffusion models and normalizing flows; and 3) a synthetic dataset of 510,000 samples (51 styles $\times$ 10,000 content samples) was curated to simulate disentanglement through systematic style-content pairing. Beyond controllable generation tasks, we demonstrate that SCFlow generalizes to ImageNet-1k and WikiArt in zero-shot settings and achieves competitive performance, highlighting that disentanglement naturally emerges from the invertible merging process. 

**Abstract (ZH)**: 通过学习可逆合并风格和内容以实现自然分离 

---
# Agentic AI in 6G Software Businesses: A Layered Maturity Model 

**Title (ZH)**: 6G软件业务中的赋能人工智能：分层成熟度模型 

**Authors**: Muhammad Zohaib, Muhammad Azeem Akbar, Sami Hyrynsalmi, Arif Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.03393)  

**Abstract**: The emergence of agentic AI systems in 6G software businesses presents both strategic opportunities and significant challenges. While such systems promise increased autonomy, scalability, and intelligent decision-making across distributed environments, their adoption raises concerns regarding technical immaturity, integration complexity, organizational readiness, and performance-cost trade-offs. In this study, we conducted a preliminary thematic mapping to identify factors influencing the adoption of agentic software within the context of 6G. Drawing on a multivocal literature review and targeted scanning, we identified 29 motivators and 27 demotivators, which were further categorized into five high-level themes in each group. This thematic mapping offers a structured overview of the enabling and inhibiting forces shaping organizational readiness for agentic transformation. Positioned as a feasibility assessment, the study represents an early phase of a broader research initiative aimed at developing and validating a layered maturity model grounded in CMMI model with the software architectural three dimensions possibly Data, Business Logic, and Presentation. Ultimately, this work seeks to provide a practical framework to help software-driven organizations assess, structure, and advance their agent-first capabilities in alignment with the demands of 6G. 

**Abstract (ZH)**: 6G软件业务中代理型AI系统的涌现既带来了战略机遇也带来了重大挑战：识别影响代理软件采用的因素 

---
# VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models via Hessian Augmentation 

**Title (ZH)**: VLMQ: 大型视觉-语言模型高效后训练量化方法及海森矩阵增强 

**Authors**: Yufei Xue, Yushi Huang, Jiawei Shao, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03351)  

**Abstract**: Post-training quantization (PTQ) has emerged as an effective approach for compressing large models and accelerating their inference without retraining. While PTQ has been extensively studied in the context of large language models (LLMs), its applicability to vision-language models (VLMs) remains underexplored. In this paper, we identify a modality discrepancy (\emph{i.e.}, limited text tokens \emph{vs.} excessive and redundant vision tokens) of VLMs. However, existing Hessian-based LLM PTQ methods treat all tokens equally during quantization, resulting in severe performance drops when applied to VLMs. Motivated by this observation, we propose a novel importance-aware PTQ framework tailored for VLMs, dubbed VLMQ. Specifically, to address vision token redundancy, VLMQ 1) optimizes an importance-aware objective that yields an enhanced Hessian with token-level importance factors, while retaining compatibility with parallelized weight updates, and 2) ensures efficiency and effectiveness by computing these factors via a single lightweight block-wise backward pass, guided by a theoretical connection to token-level perturbations. Extensive evaluations on 8 benchmarks across 0.5B$\sim$32B VLMs demonstrate the state-of-the-art (SOTA) performance of our VLMQ, particularly under low-bit settings. For example, it achieves a substantial \textbf{16.45\%} improvement on MME-RealWorld under 2-bit quantization. 

**Abstract (ZH)**: 后训练量化（PTQ）已成为一种有效的方法，可以在不重新训练的情况下压缩大型模型并加速其推理。尽管PTQ在大型语言模型（LLMs）的背景下得到了广泛研究，但其在视觉语言模型（VLMs）中的适用性仍鲜有探索。在本文中，我们识别了VLMs的模态差异（即有限的文字标记 vs. 过剩的冗余视觉标记）。然而，现有的基于海森矩阵的LLM后训练量化方法在量化过程中平等对待所有标记，这导致了在应用于VLMs时性能急剧下降。基于这一观察，我们提出了一种专为VLMs设计的新型重要性感知后训练量化框架，称为VLMQ。具体而言，为了应对视觉标记冗余，VLMQ 1) 优化了一个重要性感知的目标函数，该函数在保留与并行权重更新兼容性的前提下生成具有标记级别重要性因子的增强海森矩阵；2) 通过一个轻量级的块级反向传递计算这些因子，并利用标记级别扰动的理论连接来保证效率和有效性。在0.5B至32B VLMs上的8个基准测试中，我们的VLMQ在低比特设置下表现出最先进的性能，例如在2比特量化下，MME-RealWorld上的性能提高了显著的16.45%。 

---
# Reliable Evaluation Protocol for Low-Precision Retrieval 

**Title (ZH)**: 低精度检索的可靠评价协议 

**Authors**: Kisu Yang, Yoonna Jang, Hwanseok Jang, Kenneth Choi, Isabelle Augenstein, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03306)  

**Abstract**: Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals. 

**Abstract (ZH)**: 降低模型参数和计算的数值精度广泛用于提高检索系统的效率。然而，在低精度下计算查询与文档的相关分数时，我们会因精度降低而观察到虚假的并列现象。这引入了基于并列解决的高结果变异性，使得评估不够可靠。为应对这一问题，我们提出了一种更稳健的检索评估协议，旨在减少得分变异性，该协议包括：（1）高精度评分（HPS），将最终评分步骤提升到更高精度，以最小的计算成本解决并列候选项；（2）感知并列的检索度量（TRM），报告期望得分、范围和偏差以量化并列候选项的顺序不确定性。我们的实验在两个检索数据集上使用三种评分函数测试多个模型，证明HPS显著减少了由并列引起的不稳定性，而TRM准确恢复了预期的度量值。这一组合能够为低精度检索提供更加一致和可靠的评估系统。 

---
# Artificial Intelligence and Generative Models for Materials Discovery -- A Review 

**Title (ZH)**: 人工 Intelligence 和生成模型在材料发现中的应用——一篇综述 

**Authors**: Albertus Denny Handoko, Riko I Made  

**Link**: [PDF](https://arxiv.org/pdf/2508.03278)  

**Abstract**: High throughput experimentation tools, machine learning (ML) methods, and open material databases are radically changing the way new materials are discovered. From the experimentally driven approach in the past, we are moving quickly towards the artificial intelligence (AI) driven approach, realizing the 'inverse design' capabilities that allow the discovery of new materials given the desired properties. This review aims to discuss different principles of AI-driven generative models that are applicable for materials discovery, including different materials representations available for this purpose. We will also highlight specific applications of generative models in designing new catalysts, semiconductors, polymers, or crystals while addressing challenges such as data scarcity, computational cost, interpretability, synthesizability, and dataset biases. Emerging approaches to overcome limitations and integrate AI with experimental workflows will be discussed, including multimodal models, physics informed architectures, and closed-loop discovery systems. This review aims to provide insights for researchers aiming to harness AI's transformative potential in accelerating materials discovery for sustainability, healthcare, and energy innovation. 

**Abstract (ZH)**: 高通量实验工具、机器学习方法和开放材料数据库正从根本上改变新材料的发现方式。从过去以实验驱动的方法，我们迅速向以人工智能驱动的方法过渡，实现了根据所需性能发现新材料的“逆向设计”能力。本文综述了适用于材料发现的人工智能驱动生成模型的不同原则，包括可供使用的不同材料表示方法。同时，本文还将特别强调生成模型在设计新型催化剂、半导体、聚合物或晶体方面的具体应用，并讨论数据稀缺性、计算成本、可解释性、可合成性以及数据集偏差等挑战。还将探讨克服这些限制以及将人工智能与实验工作流程整合的新兴方法，包括多模态模型、物理信息架构和闭环发现系统。本文旨在为希望利用人工智能在可持续发展、医疗健康和能源创新中加速材料发现的研究人员提供洞见。 

---
# Approximate Proportionality in Online Fair Division 

**Title (ZH)**: 在线公平分配中的近似比例原则 

**Authors**: Davin Choo, Winston Fu, Derek Khu, Tzeh Yuan Neoh, Tze-Yang Poon, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2508.03253)  

**Abstract**: We study the online fair division problem, where indivisible goods arrive sequentially and must be allocated immediately and irrevocably to agents. Prior work has established strong impossibility results for approximating classic fairness notions, such as envy-freeness and maximin share fairness, in this setting. In contrast, we focus on proportionality up to one good (PROP1), a natural relaxation of proportionality whose approximability remains unresolved. We begin by showing that three natural greedy algorithms fail to guarantee any positive approximation to PROP1 in general, against an adaptive adversary. This is surprising because greedy algorithms are commonly used in fair division and a natural greedy algorithm is known to be able to achieve PROP1 under additional information assumptions. This hardness result motivates the study of non-adaptive adversaries and the use of side-information, in the spirit of learning-augmented algorithms. For non-adaptive adversaries, we show that the simple uniformly random allocation can achieve a meaningful PROP1 approximation with high probability. Meanwhile, we present an algorithm that obtain robust approximation ratios against PROP1 when given predictions of the maximum item value (MIV). Interestingly, we also show that stronger fairness notions such as EF1, MMS, and PROPX remain inapproximable even with perfect MIV predictions. 

**Abstract (ZH)**: 在线公平分配问题中的比例原则近似研究 

---
# The Power of Many: Synergistic Unification of Diverse Augmentations for Efficient Adversarial Robustness 

**Title (ZH)**: 多元合力：多样增强的协同统一以实现高效的对抗鲁棒性 

**Authors**: Wang Yu-Hang, Shiwei Li, Jianxiang Liao, Li Bohan, Jian Liu, Wenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03213)  

**Abstract**: Adversarial perturbations pose a significant threat to deep learning models. Adversarial Training (AT), the predominant defense method, faces challenges of high computational costs and a degradation in standard performance. While data augmentation offers an alternative path, existing techniques either yield limited robustness gains or incur substantial training overhead. Therefore, developing a defense mechanism that is both highly efficient and strongly robust is of paramount this http URL this work, we first conduct a systematic analysis of existing augmentation techniques, revealing that the synergy among diverse strategies -- rather than any single method -- is crucial for enhancing robustness. Based on this insight, we propose the Universal Adversarial Augmenter (UAA) framework, which is characterized by its plug-and-play nature and training efficiency. UAA decouples the expensive perturbation generation process from model training by pre-computing a universal transformation offline, which is then used to efficiently generate unique adversarial perturbations for each sample during this http URL experiments conducted on multiple benchmarks validate the effectiveness of UAA. The results demonstrate that UAA establishes a new state-of-the-art (SOTA) for data-augmentation-based adversarial defense strategies , without requiring the online generation of adversarial examples during training. This framework provides a practical and efficient pathway for building robust models,Our code is available in the supplementary materials. 

**Abstract (ZH)**: adversarial 水平下的高效且强健的防御机制：基于通用对抗增强的对抗训练 

---
# GeoShield: Safeguarding Geolocation Privacy from Vision-Language Models via Adversarial Perturbations 

**Title (ZH)**: GeoShield: 通过对抗性扰动保护地理定位隐私的视觉-语言模型 

**Authors**: Xinwei Liu, Xiaojun Jia, Yuan Xun, Simeng Qin, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.03209)  

**Abstract**: Vision-Language Models (VLMs) such as GPT-4o now demonstrate a remarkable ability to infer users' locations from public shared images, posing a substantial risk to geoprivacy. Although adversarial perturbations offer a potential defense, current methods are ill-suited for this scenario: they often perform poorly on high-resolution images and low perturbation budgets, and may introduce irrelevant semantic content. To address these limitations, we propose GeoShield, a novel adversarial framework designed for robust geoprivacy protection in real-world scenarios. GeoShield comprises three key modules: a feature disentanglement module that separates geographical and non-geographical information, an exposure element identification module that pinpoints geo-revealing regions within an image, and a scale-adaptive enhancement module that jointly optimizes perturbations at both global and local levels to ensure effectiveness across resolutions. Extensive experiments on challenging benchmarks show that GeoShield consistently surpasses prior methods in black-box settings, achieving strong privacy protection with minimal impact on visual or semantic quality. To our knowledge, this work is the first to explore adversarial perturbations for defending against geolocation inference by advanced VLMs, providing a practical and effective solution to escalating privacy concerns. 

**Abstract (ZH)**: Vision-Language模型（VLMs）如GPT-4o现在展示了从公共共享图像中推断用户位置的惊人能力，对地理隐私构成了重大风险。虽然对抗性扰动提供了一种潜在的防御手段，但当前的方法并不适合这种场景：它们在高分辨率图像和低扰动预算情况下往往表现不佳，并且可能会引入无关的语义内容。为了解决这些限制，我们提出了一种名为GeoShield的新型对抗框架，旨在在现实场景中实现稳健的地理隐私保护。GeoShield包含三个关键模块：一个特征解耦模块，用于分离地理信息和非地理信息；一个曝光元素识别模块，用于定位图像中的地理揭示区域；以及一个尺度自适应增强模块，用于在全局和局部两个层面上同时优化扰动，以确保在不同分辨率下均具有有效性。在具有挑战性的基准上的广泛实验表明，GeoShield在黑盒设置中始终超越了之前的方法，实现了强大的隐私保护，同时对视觉或语义质量的影响最小。据我们所知，这是首次探索对抗性扰动来防御先进VLMs的地理位置推理的方法，提供了应对不断升级的隐私担忧的实用且有效的解决方案。 

---
# Spatiotemporal wall pressure forecast of a rectangular cylinder with physics-aware DeepUFNet 

**Title (ZH)**: 具有物理意识的DeepUFNet的空间时间壁压力预测模型 

**Authors**: Junle Liu, Chang Liu, Yanyu Ke, Wenliang Chen, Kihing Shum, K.T. Tse, Gang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03183)  

**Abstract**: The wall pressure is of great importance in understanding the forces and structural responses induced by fluid. Recent works have investigated the potential of deep learning techniques in predicting mean pressure coefficients and fluctuating pressure coefficients, but most of existing deep learning frameworks are limited to predicting a single snapshot using full spatial information. To forecast spatiotemporal wall pressure of flow past a rectangular cylinder, this study develops a physics-aware DeepU-Fourier neural Network (DeepUFNet) deep learning model. DeepUFNet comprises the UNet structure and the Fourier neural network, with physical high-frequency loss control embedded in the model training stage to optimize model performance, where the parameter $\beta$ varies with the development of the training epoch. Wind tunnel testing is performed to collect wall pressures of a two-dimensional rectangular cylinder with a side ratio of 1.5 at an angle of attack of zero using high-frequency pressure scanning, thereby constructing a database for DeepUFNet training and testing. The DeepUFNet model is found to forecast spatiotemporal wall pressure information with high accuracy. The comparison between forecast results and experimental data presents agreement in statistical information, temporal pressure variation, power spectrum density, spatial distribution, and spatiotemporal correlation. It is also found that embedding a physical high-frequency loss control coefficient $\beta$ in the DeepUFNet model can significantly improve model performance in forecasting spatiotemporal wall pressure information, in particular, in forecasting high-order frequency fluctuation and wall pressure variance. Furthermore, the DeepUFNet extrapolation capability is tested with sparse spatial information input, and the model presents a satisfactory extrapolation ability 

**Abstract (ZH)**: 基于物理感知的DeepU-Fourier神经网络用于预测流经矩形柱的时空壁面压力 

---
# StoryEnsemble: Enabling Dynamic Exploration & Iteration in the Design Process with AI and Forward-Backward Propagation 

**Title (ZH)**: StoryEnsemble：通过AI和前向-后向传播实现设计过程中的动态探索与迭代 

**Authors**: Sangho Suh, Michael Lai, Kevin Pu, Steven P. Dow, Tovi Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2508.03182)  

**Abstract**: Design processes involve exploration, iteration, and movement across interconnected stages such as persona creation, problem framing, solution ideation, and prototyping. However, time and resource constraints often hinder designers from exploring broadly, collecting feedback, and revisiting earlier assumptions-making it difficult to uphold core design principles in practice. To better understand these challenges, we conducted a formative study with 15 participants-comprised of UX practitioners, students, and instructors. Based on the findings, we developed StoryEnsemble, a tool that integrates AI into a node-link interface and leverages forward and backward propagation to support dynamic exploration and iteration across the design process. A user study with 10 participants showed that StoryEnsemble enables rapid, multi-directional iteration and flexible navigation across design stages. This work advances our understanding of how AI can foster more iterative design practices by introducing novel interactions that make exploration and iteration more fluid, accessible, and engaging. 

**Abstract (ZH)**: 设计过程涉及探索、迭代和在 personas 创建、问题界定、解决方案构思和原型制作等相互连接的阶段之间移动。然而，时间与资源限制常常阻碍设计师广泛探索、收集反馈并重访早期假设，使其难以在实际设计过程中坚守核心设计原则。为了更好地理解这些挑战，我们对15名参与者（包括用户体验从业者、学生和教师）进行了形成性研究。基于研究发现，我们开发了StoryEnsemble工具，该工具将AI集成到节点链接接口中，并利用正向和反向传播支持设计过程中动态探索和迭代。用户研究显示，StoryEnsemble能够促进快速、多向迭代并在设计阶段之间实现灵活导航。本研究通过引入使探索和迭代更加流畅、易获取和引人入胜的新型交互方式，推进了我们对AI如何促进更迭代设计实践的理解。 

---
# NANDA Adaptive Resolver: Architecture for Dynamic Resolution of AI Agent Names 

**Title (ZH)**: NANDA自适应解析器：动态解决AI代理名称的架构 

**Authors**: John Zinky, Hema Seshadri, Mahesh Lambe, Pradyumna Chari, Ramesh Raskar  

**Link**: [PDF](https://arxiv.org/pdf/2508.03113)  

**Abstract**: AdaptiveResolver is a dynamic microservice architecture designed to address the limitations of static endpoint resolution for AI agent communication in distributed, heterogeneous environments. Unlike traditional DNS or static URLs, AdaptiveResolver enables context-aware, real-time selection of communication endpoints based on factors such as geographic location, system load, agent capabilities, and security threats. Agents advertise their Agent Name and context requirements through Agent Fact cards in an Agent Registry/Index. A requesting Agent discovers a Target Agent using the registry. The Requester Agent can then resolve the Target Agent Name to obtain a tailored communication channel to the agent based on actual environmental context between the agents. The architecture supports negotiation of trust, quality of service, and resource constraints, facilitating flexible, secure, and scalable agent-to-agent interactions that go beyond the classic client-server model. AdaptiveResolver provides a foundation for robust, future-proof agent communication that can evolve with increasing ecosystem complexity. 

**Abstract (ZH)**: 自适应解析器是一种动态微服务架构，旨在解决分布式异构环境中人工智能代理通信的静态端点解析限制。自适应解析器根据地理位置、系统负载、代理能力及安全威胁等因素实现上下文感知的实时端点选择，不同于传统的DNS或静态URL。代理通过代理事实卡片在代理注册表/索引中宣传其代理名称和上下文要求，请求代理使用注册表发现目标代理，然后可以根据代理间实际的环境上下文解析目标代理名称以获得量身定制的通信通道。该架构支持信任、服务质量及资源约束的协商，促进超越经典客户端-服务器模型的灵活、安全和可扩展的代理间交互。自适应解析器为具有鲁棒性和未来适应性的代理通信提供了基础，能够随着生态系统复杂性的增加而演进。 

---
# GEDAN: Learning the Edit Costs for Graph Edit Distance 

**Title (ZH)**: GEDAN: 学习图编辑距离的编辑成本 

**Authors**: Francesco Leonardi, Markus Orsi, Jean-Louis Reymond, Kaspar Riesen  

**Link**: [PDF](https://arxiv.org/pdf/2508.03111)  

**Abstract**: Graph Edit Distance (GED) is defined as the minimum cost transformation of one graph into another and is a widely adopted metric for measuring the dissimilarity between graphs. The major problem of GED is that its computation is NP-hard, which has in turn led to the development of various approximation methods, including approaches based on neural networks (NN). Most of these NN-based models simplify the problem of GED by assuming unit-cost edit operations, a rather unrealistic constraint in real-world applications. In this work, we present a novel Graph Neural Network framework that approximates GED using both supervised and unsupervised training. In the unsupervised setting, it employs a gradient-only self-organizing mechanism that enables optimization without ground-truth distances. Moreover, a core component of our architecture is the integration of a Generalized Additive Model, which allows the flexible and interpretable learning of context-aware edit costs. Experimental results show that the proposed method achieves similar results as state-of-the-art reference methods, yet significantly improves both adaptability and interpretability. That is, the learned cost function offers insights into complex graph structures, making it particularly valuable in domains such as molecular analysis and structural pattern discovery. 

**Abstract (ZH)**: Graph Edit Distance (GED)，编辑距离（GED）定义为将一个图转换为另一个图的最小花费变换， GED是衡量图之间差异性的广泛采用的度度度度量。GED的主要问题是其计算是NP难问题，导致了各种近近近变换方法的发展，包括包括包括基于神经网络（NN的方法。G大部分基于NN的模型基于GED问题基于单一成本变换操作的约束G这个约束在实际应用中显得相当不合理。G我们提出了一种G G G G基于图的神经网络框架，通过监督和无监督训练近近近近G G G G计算GED。G该无监督部分采用纯梯度度度度度度流优化机制，能够优化真实距离。G此外G G G G网络架构结合 G整合了加一种基于上下文感知的灵活可且可可 G可可可 G G可 G G G可可解释的编辑编辑代价。实验表明该方法能达到与其 G G G G当前最先进的方法相当G并在适应性和解释性 G G G G性 G G G G G性 G提供了对于复杂图结构的见解G在诸如分子分析和和结构发现等 G G G G G等 G等领域特别有价值。 

---
# Pseudo-label Induced Subspace Representation Learning for Robust Out-of-Distribution Detection 

**Title (ZH)**: 伪标签引导的子空间表示学习以实现稳健的分布外检测 

**Authors**: Tarhib Al Azad, Faizul Rakib Sayem, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03108)  

**Abstract**: Out-of-distribution (OOD) detection lies at the heart of robust artificial intelligence (AI), aiming to identify samples from novel distributions beyond the training set. Recent approaches have exploited feature representations as distinguishing signatures for OOD detection. However, most existing methods rely on restrictive assumptions on the feature space that limit the separability between in-distribution (ID) and OOD samples. In this work, we propose a novel OOD detection framework based on a pseudo-label-induced subspace representation, that works under more relaxed and natural assumptions compared to existing feature-based techniques. In addition, we introduce a simple yet effective learning criterion that integrates a cross-entropy-based ID classification loss with a subspace distance-based regularization loss to enhance ID-OOD separability. Extensive experiments validate the effectiveness of our framework. 

**Abstract (ZH)**: 异分布（OOD）检测是稳健人工智能的核心，旨在识别训练集之外的新分布样本。近期方法利用特征表示作为异分布检测的区分特征。然而，现有大多数方法依赖于特征空间的严格假设，这限制了内分布（ID）和异分布样本之间的可分性。在本工作中，我们提出了一种基于伪标签诱导子空间表示的新颖OOD检测框架，该框架在比现有特征基技术更为宽松和自然的假设下工作。此外，我们引入了一种简单而有效的学习准则，通过结合基于交叉熵的内分布分类损失与基于子空间距离的正则化损失，增强内分布与异分布样本之间的可分性。 extensive实验验证了我们框架的有效性。 

---
# HiTeC: Hierarchical Contrastive Learning on Text-Attributed Hypergraph with Semantic-Aware Augmentation 

**Title (ZH)**: HiTeC: 带有语义增强的文本归因超图分层对比学习 

**Authors**: Mengting Pan, Fan Li, Xiaoyang Wang, Wenjie Zhang, Xuemin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03104)  

**Abstract**: Contrastive learning (CL) has become a dominant paradigm for self-supervised hypergraph learning, enabling effective training without costly labels. However, node entities in real-world hypergraphs are often associated with rich textual information, which is overlooked in prior works. Directly applying existing CL-based methods to such text-attributed hypergraphs (TAHGs) leads to three key limitations: (1) The common use of graph-agnostic text encoders overlooks the correlations between textual content and hypergraph topology, resulting in suboptimal representations. (2) Their reliance on random data augmentations introduces noise and weakens the contrastive objective. (3) The primary focus on node- and hyperedge-level contrastive signals limits the ability to capture long-range dependencies, which is essential for expressive representation learning. Although HyperBERT pioneers CL on TAHGs, its co-training paradigm suffers from poor scalability. To fill the research gap, we introduce HiTeC, a two-stage hierarchical contrastive learning framework with semantic-aware augmentation for scalable and effective self-supervised learning on TAHGs. In the first stage, we pre-train the text encoder with a structure-aware contrastive objective to overcome the graph-agnostic nature of conventional methods. In the second stage, we introduce two semantic-aware augmentation strategies, including prompt-enhanced text augmentation and semantic-aware hyperedge drop, to facilitate informative view generation. Furthermore, we propose a multi-scale contrastive loss that extends existing objectives with an $s$-walk-based subgraph-level contrast to better capture long-range dependencies. By decoupling text encoder pretraining from hypergraph contrastive learning, this two-stage design enhances scalability without compromising representation quality. Extensive experiments confirm the effectiveness of HiTeC. 

**Abstract (ZH)**: 基于语义感知的两阶段层次对比学习框架HiTeC：面向文本标注超图的 scalable 自监督学习 

---
# Using the NANDA Index Architecture in Practice: An Enterprise Perspective 

**Title (ZH)**: 在实践中应用NANDA索引架构：从企业视角出发 

**Authors**: Sichao Wang, Ramesh Raskar, Mahesh Lambe, Pradyumna Chari, Rekha Singhal, Shailja Gupta, Rajesh Ranjan, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03101)  

**Abstract**: The proliferation of autonomous AI agents represents a paradigmatic shift from traditional web architectures toward collaborative intelligent systems requiring sophisticated mechanisms for discovery, authentication, capability verification, and secure collaboration across heterogeneous protocol environments. This paper presents a comprehensive framework addressing the fundamental infrastructure requirements for secure, trustworthy, and interoperable AI agent ecosystems. We introduce the NANDA (Networked AI Agents in a Decentralized Architecture) framework, providing global agent discovery, cryptographically verifiable capability attestation through AgentFacts, and cross-protocol interoperability across Anthropic's Modal Context Protocol (MCP), Google's Agent-to-Agent (A2A), Microsoft's NLWeb, and standard HTTPS communications. NANDA implements Zero Trust Agentic Access (ZTAA) principles, extending traditional Zero Trust Network Access (ZTNA) to address autonomous agent security challenges including capability spoofing, impersonation attacks, and sensitive data leakage. The framework defines Agent Visibility and Control (AVC) mechanisms enabling enterprise governance while maintaining operational autonomy and regulatory compliance. Our approach transforms isolated AI agents into an interconnected ecosystem of verifiable, trustworthy intelligent services, establishing foundational infrastructure for large-scale autonomous agent deployment across enterprise and consumer environments. This work addresses the critical gap between current AI agent capabilities and infrastructure requirements for secure, scalable, multi-agent collaboration, positioning the foundation for next-generation autonomous intelligent systems. 

**Abstract (ZH)**: 自主人工智能代理的激增代表了从传统网络架构向需要复杂发现、认证、能力验证和跨异构协议环境安全协作的协作智能系统范式转变。本文提出了一种全面的框架，以满足安全、可信和互操作的人工智能代理生态系统的基本基础设施要求。我们介绍了NANDA（去中心化架构中的网络人工智能代理）框架，提供全球代理发现、通过AgentFacts进行加密可验证的能力证明，并在Anthropic的Modal Context Protocol (MCP)、Google的Agent-to-Agent (A2A)、Microsoft的NLWeb以及标准HTTPS通信之间实现跨协议互操作性。NANDA 实现了零信任代理访问 (ZTAA) 原则，将传统的零信任网络访问 (ZTNA) 扩展到应对自主代理安全挑战，包括能力欺骗、冒充攻击和敏感数据泄漏。该框架定义了代理可见性和控制 (AVC) 机制，允许企业治理同时保持操作自主性和合规性。我们的方法将孤立的人工智能代理转变为一个互连的、可验证和可信的智能服务生态系统，为大规模自主代理在企业和消费者环境中的部署奠定了基础。本研究填补了当前人工智能代理能力和安全、可扩展、多代理协作基础设施需求之间的关键缺口，为新一代自主智能系统奠定了基础。 

---
# A Survey of AI Agent Registry Solutions 

**Title (ZH)**: AI代理注册解决方案综述 

**Authors**: Aditi Singh, Abul Ehtesham, Ramesh Raskar, Mahesh Lambe, Pradyumna Chari, Jared James Grogan, Abhishek Singh, Saket Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.03095)  

**Abstract**: As As autonomous AI agents scale across cloud, enterprise, and decentralized environments, the need for standardized registry systems to support discovery, identity, and capability sharing has become essential. This paper surveys three prominent registry approaches each defined by a unique metadata model: MCP's this http URL, A2A's Agent Card, and NANDA's AgentFacts. MCP uses a centralized metaregistry with GitHub authenticated publishing and structured metadata for server discovery. A2A enables decentralized interaction via JSON-based Agent Cards, discoverable through well-known URIs, curated catalogs, or direct configuration. NANDA Index introduces AgentFacts, a cryptographically verifiable and privacy-preserving metadata model designed for dynamic discovery, credentialed capabilities, and cross-domain interoperability. These approaches are compared across four dimensions: security, scalability, authentication, and maintainability. The paper concludes with suggestions and recommendations to guide future design and adoption of registry systems for the Internet of AI Agents. 

**Abstract (ZH)**: 随着自主AI代理在云、企业及分散环境中扩展，建立支持发现、标识和能力共享的标准注册系统的需求变得至关重要。本文综述了三种独特的元数据模型定义的注册方法：MCP的this http URL、A2A的Agent Card以及NANDA的AgentFacts。MCP使用一个集中式的元注册系统，通过GitHub认证发布并使用结构化元数据进行服务器发现。A2A通过基于JSON的Agent Card实现去中心化的交互，可以通过已知的URI、精心策划的目录或直接配置进行发现。NANDA Index引入了AgentFacts，这是一种可通过密码学验证且保护隐私的元数据模型，适用于动态发现、凭据授权能力和跨域互操作性。本文从四个维度——安全性、可扩展性、认证和可维护性——比较了这些方法。论文最后提出了建议和建议，以指导未来AI代理互联网注册系统的设计和采用。 

---
# Untraceable DeepFakes via Traceable Fingerprint Elimination 

**Title (ZH)**: 可追踪水印消除下的不可追溯深度伪生成 

**Authors**: Jiewei Lai, Lan Zhang, Chen Tang, Pengcheng Sun, Xinming Wang, Yunhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03067)  

**Abstract**: Recent advancements in DeepFakes attribution technologies have significantly enhanced forensic capabilities, enabling the extraction of traces left by generative models (GMs) in images, making DeepFakes traceable back to their source GMs. Meanwhile, several attacks have attempted to evade attribution models (AMs) for exploring their limitations, calling for more robust AMs. However, existing attacks fail to eliminate GMs' traces, thus can be mitigated by defensive measures. In this paper, we identify that untraceable DeepFakes can be achieved through a multiplicative attack, which can fundamentally eliminate GMs' traces, thereby evading AMs even enhanced with defensive measures. We design a universal and black-box attack method that trains an adversarial model solely using real data, applicable for various GMs and agnostic to AMs. Experimental results demonstrate the outstanding attack capability and universal applicability of our method, achieving an average attack success rate (ASR) of 97.08\% against 6 advanced AMs on DeepFakes generated by 9 GMs. Even in the presence of defensive mechanisms, our method maintains an ASR exceeding 72.39\%. Our work underscores the potential challenges posed by multiplicative attacks and highlights the need for more robust AMs. 

**Abstract (ZH)**: Recent advancements in DeepFakes attribution technologies have significantly enhanced forensic capabilities, enabling the extraction of traces left by generative models (GMs) in images, making DeepFakes traceable back to their source GMs. Meanwhile, several attacks have attempted to evade attribution models (AMs) for exploring their limitations, calling for more robust AMs. However, existing attacks fail to eliminate GMs' traces, thus can be mitigated by defensive measures. In this paper, we identify that untraceable DeepFakes can be achieved through a multiplicative attack, which can fundamentally eliminate GMs' traces, thereby evading AMs even enhanced with defensive measures. We design a universal and black-box attack method that trains an adversarial model solely using real data, applicable for various GMs and agnostic to AMs. Experimental results demonstrate the outstanding attack capability and universal applicability of our method, achieving an average attack success rate (ASR) of 97.08% against 6 advanced AMs on DeepFakes generated by 9 GMs. Even in the presence of defensive mechanisms, our method maintains an ASR exceeding 72.39%. Our work underscores the potential challenges posed by multiplicative attacks and highlights the need for more robust AMs. 

---
# VRPO: Rethinking Value Modeling for Robust RL Training under Noisy Supervision 

**Title (ZH)**: VRPO: 重新思考在嘈杂监督下的稳健 reinforcement learning 训练的价值建模 

**Authors**: Dingwei Zhu, Shihan Dou, Zhiheng Xi, Senjie Jin, Guoqiang Zhang, Jiazheng Zhang, Junjie Ye, Mingxu Chai, Enyu Zhou, Ming Zhang, Caishuang Huang, Yunke Zhang, Yuran Wang, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2508.03058)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) often suffers from noisy or imperfect reward supervision in real-world settings, which undermines policy stability and generalization. Such noise may cause models to lose attention on key words during advantage estimation. While prior work focuses on reward denoising or filtering poor data, it often overlooks the critical role of the value model in policy optimization. In this work, we show that a strong value model is essential for mitigating noise by absorbing unstable signals and enabling more reliable advantage estimation. We propose VRPO, a value-centric framework for robust PPO training under noisy supervision. VRPO combines two core designs: (1) an auxiliary loss guided by entropy and perplexity from a frozen language model, and (2) a variational information bottleneck. These mechanisms enhance the value model's ability to filter out noise and capture key words from the context during advantage estimation, transforming it from a passive predictor into an active regulator of noise. Experiments on math reasoning, science QA, and multi-turn dialogue, under both rule-based and model-based noisy rewards, show that VRPO consistently outperforms PPO and GRPO baselines. Our findings underscore the often-overlooked importance of the value model in RLHF and offer a principled and practical approach to robust policy optimization in noisy real-world environments. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）中的噪声或 imperfect 奖励监督往往损害策略的稳定性和泛化能力，这削弱了政策的有效性。此类噪声可能导致模型在优势估计时忽视关键词汇。尽管先前的工作集中在奖励去噪或过滤不良数据上，但往往忽略了价值模型在策略优化中的关键作用。在本文中，我们展示了强大价值模型对于通过吸收不稳定信号并促进更可靠的优势估计来减轻噪声的重要性。我们提出了一种基于价值的框架 VRPO，以在嘈杂监督下进行鲁棒的 PPO 训练。VRPO 结合了两个核心设计：（1）由冻结语言模型的熵和困惑度引导的辅助损失，以及（2）变异信息瓶颈。这些机制增强了价值模型在优势估计过程中过滤噪声并从上下文中捕获关键词汇的能力，使其从被动预测器转变为噪声的主动调节器。在数学推理、科学问答和多轮对话任务中，无论是基于规则的还是基于模型的噪声奖励环境，实验结果均表明 VRPO 一贯优于 PPO 和 GRPO 基线。我们的研究强调了在 RLHF 中被忽视的价值模型的重要性，并提出了在嘈杂的现实环境中进行鲁棒策略优化的原理性方法。 

---
# Autonomous Inorganic Materials Discovery via Multi-Agent Physics-Aware Scientific Reasoning 

**Title (ZH)**: 自主物理意识科学推理的多代理无机材料发现 

**Authors**: Alireza Ghafarollahi, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2508.02956)  

**Abstract**: Conventional machine learning approaches accelerate inorganic materials design via accurate property prediction and targeted material generation, yet they operate as single-shot models limited by the latent knowledge baked into their training data. A central challenge lies in creating an intelligent system capable of autonomously executing the full inorganic materials discovery cycle, from ideation and planning to experimentation and iterative refinement. We introduce SparksMatter, a multi-agent AI model for automated inorganic materials design that addresses user queries by generating ideas, designing and executing experimental workflows, continuously evaluating and refining results, and ultimately proposing candidate materials that meet the target objectives. SparksMatter also critiques and improves its own responses, identifies research gaps and limitations, and suggests rigorous follow-up validation steps, including DFT calculations and experimental synthesis and characterization, embedded in a well-structured final report. The model's performance is evaluated across case studies in thermoelectrics, semiconductors, and perovskite oxides materials design. The results demonstrate the capacity of SparksMatter to generate novel stable inorganic structures that target the user's needs. Benchmarking against frontier models reveals that SparksMatter consistently achieves higher scores in relevance, novelty, and scientific rigor, with a significant improvement in novelty across multiple real-world design tasks as assessed by a blinded evaluator. These results demonstrate SparksMatter's unique capacity to generate chemically valid, physically meaningful, and creative inorganic materials hypotheses beyond existing materials knowledge. 

**Abstract (ZH)**: 基于机器学习的常规方法通过准确的性质预测和针对性的材料生成加速了无机材料的设计，但它们作为单一-shot模型受到训练数据中潜藏知识的限制。一个核心挑战是创建一个能够自主执行完整的无机材料发现循环（从构想到规划、实验以及迭代优化）的智能系统。我们介绍了一种多智能体AI模型——SparksMatter，该模型通过生成想法、设计和执行实验工作流程、持续评估和优化结果，并最终提出满足目标要求的候选材料来响应用户查询。SparksMatter 也会批判性地改进自己的回应，并识别研究缺口和限制，建议严格的后续验证步骤，包括DFT计算和实验合成及表征，嵌入到结构良好的最终报告中。该模型的表现通过对热电材料、半导体和钙钛矿氧化物材料设计案例研究进行了评估。结果表明，SparksMatter 能够生成符合用户需求的新型稳定无机结构。与前沿模型基准测试表明，SparksMatter 在相关性、创新性和科学严谨性方面始终获得更高分数，并在多个实际设计任务中表现出显著的创新性改进，由盲评员评估。这些结果展示了 SparksMatter 拥有超越现有材料知识生成化学上有效、物理上有意义和创新性的无机材料假设的独特能力。 

---
# Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment 

**Title (ZH)**: 在推荐系统中实现标度律：一种超大规模模型部署的基石-专家 paradigma 

**Authors**: Dai Li, Kevin Course, Wei Li, Hongwei Li, Jie Hua, Yiqi Chen, Zhao Zhu, Rui Jian, Xuan Cao, Bi Xue, Yu Shi, Jing Qian, Kai Ren, Matt Ma, Qunshu Zhang, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02929)  

**Abstract**: While scaling laws promise significant performance gains for recommender systems, efficiently deploying hyperscale models remains a major unsolved challenge. In contrast to fields where FMs are already widely adopted such as natural language processing and computer vision, progress in recommender systems is hindered by unique challenges including the need to learn from online streaming data under shifting data distributions, the need to adapt to different recommendation surfaces with a wide diversity in their downstream tasks and their input distributions, and stringent latency and computational constraints. To bridge this gap, we propose to leverage the Foundation-Expert Paradigm: a framework designed for the development and deployment of hyperscale recommendation FMs. In our approach, a central FM is trained on lifelong, cross-surface, multi-modal user data to learn generalizable knowledge. This knowledge is then efficiently transferred to various lightweight, surface-specific ``expert" models via target-aware embeddings, allowing them to adapt to local data distributions and optimization goals with minimal overhead. To meet our training, inference and development needs, we built HyperCast, a production-grade infrastructure system that re-engineers training, serving, logging and iteration to power this decoupled paradigm. Our approach is now deployed at Meta serving tens of billions of user requests daily, demonstrating online metric improvements over our previous one-stage production system while improving developer velocity and maintaining infrastructure efficiency. To the best of our knowledge, this work represents the first successful deployment of a Foundation-Expert paradigm at this scale, offering a proven, compute-efficient, and developer-friendly blueprint to realize the promise of scaling laws in recommender systems. 

**Abstract (ZH)**: 基于基础模型-专家模型范式的大型推荐系统高效部署：克服独特挑战并实现性能提升 

---
# Engineered over Emergent Communication in MARL for Scalable and Sample-Efficient Cooperative Task Allocation in a Partially Observable Grid 

**Title (ZH)**: 工程化的 emergent 通信在部分可观测网格多智能体强化学习中的应用：面向可扩展和样本高效的合作任务分配 

**Authors**: Brennen A. Hill, Mant Koh En Wei, Thangavel Jishnuanandh  

**Link**: [PDF](https://arxiv.org/pdf/2508.02912)  

**Abstract**: We compare the efficacy of learned versus engineered communication strategies in a cooperative multi-agent reinforcement learning (MARL) environment. For the learned approach, we introduce Learned Direct Communication (LDC), where agents generate messages and actions concurrently via a neural network. Our engineered approach, Intention Communication, employs an Imagined Trajectory Generation Module (ITGM) and a Message Generation Network (MGN) to formulate messages based on predicted future states. Both strategies are evaluated on their success rates in cooperative tasks under fully and partially observable conditions. Our findings indicate that while emergent communication is viable, the engineered approach demonstrates superior performance and scalability, particularly as environmental complexity increases. 

**Abstract (ZH)**: 我们在合作多智能体强化学习（MARL）环境中比较了学习获得与工程设计的通信策略的有效性。对于学习获得的方法，我们引入了直接通信学习（LDC），其中智能体通过神经网络同时生成消息和行动。我们设计的方法，意图通信，使用想象的轨迹生成模块（ITGM）和消息生成网络（MGN）根据预测的未来状态制定消息。两种策略都在完全可观测和部分可观测条件下合作任务的成功率上进行了评估。我们的研究表明，虽然 Emergent 通信是可行的，但工程设计的方法在性能和扩展性上更优，尤其是在环境复杂性增加时更为明显。 

---
# CauKer: classification time series foundation models can be pretrained on synthetic data only 

**Title (ZH)**: CauKer: 分类时间序列基础模型仅可在合成数据上预训练 

**Authors**: Shifeng Xie, Vasilii Feofanov, Marius Alonso, Ambroise Odonnat, Jianfeng Zhang, Themis Palpanas, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2508.02879)  

**Abstract**: Time series foundation models (TSFMs) have recently gained significant attention due to their strong zero-shot capabilities and widespread real-world applications. Such models typically require a computationally costly pretraining on large-scale, carefully curated collections of real-world sequences. To allow for a sample-efficient pretraining of TSFMs, we propose CauKer, a novel algorithm designed to generate diverse, causally coherent synthetic time series with realistic trends, seasonality, and nonlinear interactions. CauKer combines Gaussian Process (GP) kernel composition with Structural Causal Models (SCM) to produce data for sample-efficient pretraining of state-of-the-art classification TSFMs having different architectures and following different pretraining approaches. Additionally, our experiments reveal that CauKer-generated datasets exhibit clear scaling laws for both dataset size (10K to 10M samples) and model capacity (1M to 783M parameters), unlike real-world datasets, which display irregular scaling behavior. 

**Abstract (ZH)**: TSFMs的时间序列基础模型近年来由于其强大的零-shot能力及广泛的实际应用而引起了大量关注。为了进行TSFMs的样本高效预训练，我们提出了一种名为CauKer的新算法，该算法旨在生成具有真实趋势、季节性和非线性交互的多样化且因果一致的合成时间序列。CauKer结合了高斯过程核函数组合与结构因果模型（SCM），用于为不同架构和预训练方法的状态-of-艺术分类TSFMs进行高效抽样预训练数据生成。此外，我们的实验表明，CauKer生成的数据集在数据集大小（10K到10M样本）和模型容量（1M到783M参数）方面表现出明确的标度定律，而现实世界数据集则表现出不规则的标度行为。 

---
# Beyond Least Squares: Robust Regression Transformer (R2T) 

**Title (ZH)**: 超越最小二乘法：稳健回归变换器(R2T) 

**Authors**: Roman Gutierrez, Tony Kai Tang, Isabel Gutierrez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02874)  

**Abstract**: Robust regression techniques rely on least-squares optimization, which works well for Gaussian noise but fails in the presence of asymmetric structured noise. We propose a hybrid neural-symbolic architecture where a transformer encoder processes numerical sequences, a compression NN predicts symbolic parameters, and a fixed symbolic equation reconstructs the original sequence. Using synthetic data, the training objective is to recover the original sequence after adding asymmetric structured noise, effectively learning a symbolic fit guided by neural parameter estimation. Our model achieves a median regression MSE of 6e-6 to 3.5e-5 on synthetic wearable data, which is a 10-300 times improvement when compared with ordinary least squares fit and robust regression techniques such as Huber loss or SoftL1. 

**Abstract (ZH)**: 鲁棒回归技术依赖于最小二乘优化，这在高斯噪声情况下表现良好，但在存在非对称结构噪声时会失效。我们提出了一种混合神经-符号架构，其中变压器编码器处理数字序列，压缩神经网络预测符号参数，固定符号方程恢复原始序列。使用合成数据，训练目标是在添加非对称结构噪声后恢复原始序列，从而有效学习由神经参数估计引导的符号拟合。我们的模型在合成穿戴设备数据上的中位数回归均方误差为6e-6至3.5e-5，与普通最小二乘拟合和Huber损失或SoftL1等鲁棒回归技术相比，改进幅度为10至300倍。 

---
# SecoustiCodec: Cross-Modal Aligned Streaming Single-Codecbook Speech Codec 

**Title (ZH)**: 跨模态对齐的流式单码率语音编解码器 

**Authors**: Chunyu Qiang, Haoyu Wang, Cheng Gong, Tianrui Wang, Ruibo Fu, Tao Wang, Ruilong Chen, Jiangyan Yi, Zhengqi Wen, Chen Zhang, Longbiao Wang, Jianwu Dang, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02849)  

**Abstract**: Speech codecs serve as a crucial bridge in unifying speech and text language models. Existing codec methods face several challenges in semantic encoding, such as residual paralinguistic information (e.g., timbre, emotion), insufficient semantic completeness, limited reconstruction capability, and lack of support for streaming. To address these challenges, we propose SecoustiCodec, a cross-modal aligned low-bitrate streaming speech codec that disentangles semantic and paralinguistic information in a single-codebook space. To ensure semantic completeness and reconstruction fidelity, paralinguistic encoding is introduced to bridge the information gap between semantic and acoustic encoding. A semantic-only efficient quantization method based on VAE (Variational Autoencoder) and FSQ (Finite Scalar Quantization) is proposed. This approach alleviates the long-tail distribution problem of tokens while maintaining high codebook utilization. A semantic disentanglement method based on contrastive learning is proposed, which aligns text and speech in a joint multimodal frame-level space, effectively removing paralinguistic information from semantic encoding. An acoustic-constrained multi-stage optimization strategy is proposed to ensure robust and stable convergence. Figure~\ref{fig:pesq_kbps_below_2kbps} shows SecoustiCodec achieves SOTA (state-of-the-art) reconstruction quality (PESQ) of 1.77/2.58 at 0.27/1 kbps. The code and model weights for SecoustiCodec will be open-sourced upon the completion of the peer-review process. We've open-sourced SecoustiCodec's demo, code, and model weights. 

**Abstract (ZH)**: 论文标题翻译如下：

语音编解码器充当连接语音和文本语言模型的关键
user
语音编解码器充当连接
user
语音编解码器充当连接语音和文本语言模型之间的重要桥梁。现有的编解码器在语义编码方面存在一些挑战，禁止输出过多空格
/Formalized the challenges in semantic encoding include the existing codecs include,这样才能限制输出的空格？
我可以这样翻译吗：语音编解码器充当连接语音和文本语言模型之间的桥梁。现有的编解码器在语义编码方面存在一些挑战。 

---
# Learning from B Cell Evolution: Adaptive Multi-Expert Diffusion for Antibody Design via Online Optimization 

**Title (ZH)**: 从B细胞进化中学习：基于在线优化的自适应多专家扩散抗体设计 

**Authors**: Hanqi Feng, Peng Qiu, Mengchun Zhang, Yiran Tao, You Fan, Jingtao Xu, Barnabas Poczos  

**Link**: [PDF](https://arxiv.org/pdf/2508.02834)  

**Abstract**: Recent advances in diffusion models have shown remarkable potential for antibody design, yet existing approaches apply uniform generation strategies that cannot adapt to each antigen's unique requirements. Inspired by B cell affinity maturation, where antibodies evolve through multi-objective optimization balancing affinity, stability, and self-avoidance, we propose the first biologically-motivated framework that leverages physics-based domain knowledge within an online meta-learning system. Our method employs multiple specialized experts (van der Waals, molecular recognition, energy balance, and interface geometry) whose parameters evolve during generation based on iterative feedback, mimicking natural antibody refinement cycles. Instead of fixed protocols, this adaptive guidance discovers personalized optimization strategies for each target. Our experiments demonstrate that this approach: (1) discovers optimal SE(3)-equivariant guidance strategies for different antigen classes without pre-training, preserving molecular symmetries throughout optimization; (2) significantly enhances hotspot coverage and interface quality through target-specific adaptation, achieving balanced multi-objective optimization characteristic of therapeutic antibodies; (3) establishes a paradigm for iterative refinement where each antibody-antigen system learns its unique optimization profile through online evaluation; (4) generalizes effectively across diverse design challenges, from small epitopes to large protein interfaces, enabling precision-focused campaigns for individual targets. 

**Abstract (ZH)**: 近年来，扩散模型在抗体设计方面的进展显示了巨大的潜力，但现有方法采用统一的生成策略，无法适应每种抗原的独特要求。受B细胞亲和力成熟机制的启发，其中抗体通过多目标优化平衡亲和力、稳定性和自避免性而进化，我们提出了第一个基于生物学动机的框架，该框架在在线元学习系统中结合了基于物理领域的专业知识。该方法采用多个专门的专家（范德瓦尔斯力、分子识别、能量平衡和界面几何），其参数在生成过程中基于迭代反馈进行演化，模拟天然抗体精细调整的周期。与固定协议不同，这种适应性指导发现每种靶标的个性化优化策略。我们的实验显示，该方法：（1）在无需预训练的情况下，为不同的抗原类别发现最优的SE(3)对称性保持的引导策略，优化过程中保持分子对称性；（2）通过靶标特定的适应性显著增强热点覆盖和界面质量，实现类似于治疗性抗体的多目标优化平衡；（3）建立了迭代细化的范式，每个抗体-抗原系统通过在线评估学习其独特的优化特征；（4）在从小表位到大蛋白质界面的多样设计挑战中表现出有效的一般性，为单个靶标实现精准定向的计划。 

---
# TransAM: Transformer-Based Agent Modeling for Multi-Agent Systems via Local Trajectory Encoding 

**Title (ZH)**: 基于局部轨迹编码的Transformer代理建模方法：多代理系统中的应用 

**Authors**: Conor Wallace, Umer Siddique, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02826)  

**Abstract**: Agent modeling is a critical component in developing effective policies within multi-agent systems, as it enables agents to form beliefs about the behaviors, intentions, and competencies of others. Many existing approaches assume access to other agents' episodic trajectories, a condition often unrealistic in real-world applications. Consequently, a practical agent modeling approach must learn a robust representation of the policies of the other agents based only on the local trajectory of the controlled agent. In this paper, we propose \texttt{TransAM}, a novel transformer-based agent modeling approach to encode local trajectories into an embedding space that effectively captures the policies of other agents. We evaluate the performance of the proposed method in cooperative, competitive, and mixed multi-agent environments. Extensive experimental results demonstrate that our approach generates strong policy representations, improves agent modeling, and leads to higher episodic returns. 

**Abstract (ZH)**: 基于Transformer的局部轨迹编码剂模型方法TransAM及其在多agent环境中的应用研究 

---
# Real-World Receptivity to Adaptive Mental Health Interventions: Findings from an In-the-Wild Study 

**Title (ZH)**: 现实世界中对适应性心理健康干预的接受度：一项野外研究的发现 

**Authors**: Nilesh Kumar Sahu, Aditya Sneh, Snehil Gupta, Haroon R Lone  

**Link**: [PDF](https://arxiv.org/pdf/2508.02817)  

**Abstract**: The rise of mobile health (mHealth) technologies has enabled real-time monitoring and intervention for mental health conditions using passively sensed smartphone data. Building on these capabilities, Just-in-Time Adaptive Interventions (JITAIs) seek to deliver personalized support at opportune moments, adapting to users' evolving contexts and needs. Although prior research has examined how context affects user responses to generic notifications and general mHealth messages, relatively little work has explored its influence on engagement with actual mental health interventions. Furthermore, while much of the existing research has focused on detecting when users might benefit from an intervention, less attention has been paid to understanding receptivity, i.e., users' willingness and ability to engage with and act upon the intervention.
In this study, we investigate user receptivity through two components: acceptance(acknowledging or engaging with a prompt) and feasibility (ability to act given situational constraints). We conducted a two-week in-the-wild study with 70 students using a custom Android app, LogMe, which collected passive sensor data and active context reports to prompt mental health interventions. The adaptive intervention module was built using Thompson Sampling, a reinforcement learning algorithm. We address four research questions relating smartphone features and self-reported contexts to acceptance and feasibility, and examine whether an adaptive reinforcement learning approach can optimize intervention delivery by maximizing a combined receptivity reward. Our results show that several types of passively sensed data significantly influenced user receptivity to interventions. Our findings contribute insights into the design of context-aware, adaptive interventions that are not only timely but also actionable in real-world settings. 

**Abstract (ZH)**: 移动健康（mHealth）技术的发展使得可以通过被动采集的智能手机数据实现对心理健康状况的实时监测和干预。基于这些能力，即时适配干预（JITAIs）旨在在适当的时候提供个性化支持，并根据用户不断变化的环境和需求进行调整。尽管先前的研究已经探讨了背景如何影响用户对通用通知和一般mHealth信息的反应，但相对较少的研究考察了其对实际心理健康干预参与度的影响。此外，虽然现有研究大多集中在检测用户何时可能从干预中受益，但较少关注用户的接受性，即他们参与并采取行动的能力。

在本研究中，我们通过两个方面来考察用户接受性：接受（认可或回应提示）和可行性（在情境限制下行动的能力）。我们使用自定义的Android应用LogMe进行了为期两周的野外研究，该应用收集了被动传感器数据和主动的上下文报告以提示心理健康干预。适应性干预模块采用了Thompson Sampling（一种强化学习算法）进行构建。我们探讨了智能手机功能和自我报告的背景与接受性和可行性之间的关系，并研究了是否可以通过最大化综合接受性奖励来利用适应性强化学习方法优化干预的交付。研究结果表明，多种类型的被动感知数据显著影响了用户对干预的接受性。我们的发现为设计既及时又在实际情境中可操作的适应性干预措施提供了见解。 

---
# Extracting Range-Doppler Information of Moving Targets from Wi-Fi Channel State Information 

**Title (ZH)**: 从Wi-Fi信道状态信息中提取目标的范围-多普勒信息 

**Authors**: Jessica Sanson, Rahul C. Shah, Maximilian Pinaroc, Valerio Frascolla  

**Link**: [PDF](https://arxiv.org/pdf/2508.02799)  

**Abstract**: This paper presents, for the first time, a method to extract both range and Doppler information from commercial Wi-Fi Channel State Information (CSI) using a monostatic (single transceiver) setup. Utilizing the CSI phase in Wi-Fi sensing from a Network Interface Card (NIC) not designed for full-duplex operation is challenging due to (1) Hardware asynchronization, which introduces significant phase errors, and (2) Proximity of transmit (Tx) and receive (Rx) antennas, which creates strong coupling that overwhelms the motion signal of interest. We propose a new signal processing approach that addresses both challenges via three key innovations: Time offset cancellation, Phase alignment correction, and Tx/Rx coupling mitigation. Our method achieves cm-level accuracy in range and Doppler estimation for moving targets, validated using a commercial Intel Wi-Fi AX211 NIC. Our results show successful detection and tracking of moving objects in realistic environments, establishing the feasibility of high-precision sensing using standard Wi-Fi packet communications and off-the-shelf hardware without requiring any modification or specialized full-duplex capabilities. 

**Abstract (ZH)**: 本研究首次提出了一种利用单频段（单收发器）设置从商业Wi-Fi频道状态信息（CSI）中提取距离和多普勒信息的方法。 

---
# Web3 x AI Agents: Landscape, Integrations, and Foundational Challenges 

**Title (ZH)**: Web3与AI代理：格局、集成与基础挑战 

**Authors**: Yiming Shen, Jiashuo Zhang, Zhenzhe Shao, Wenxuan Luo, Yanlin Wang, Ting Chen, Zibin Zheng, Jiachi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02773)  

**Abstract**: The convergence of Web3 technologies and AI agents represents a rapidly evolving frontier poised to reshape decentralized ecosystems. This paper presents the first and most comprehensive analysis of the intersection between Web3 and AI agents, examining five critical dimensions: landscape, economics, governance, security, and trust mechanisms. Through an analysis of 133 existing projects, we first develop a taxonomy and systematically map the current market landscape (RQ1), identifying distinct patterns in project distribution and capitalization. Building upon these findings, we further investigate four key integrations: (1) the role of AI agents in participating in and optimizing decentralized finance (RQ2); (2) their contribution to enhancing Web3 governance mechanisms (RQ3); (3) their capacity to strengthen Web3 security via intelligent vulnerability detection and automated smart contract auditing (RQ4); and (4) the establishment of robust reliability frameworks for AI agent operations leveraging Web3's inherent trust infrastructure (RQ5). By synthesizing these dimensions, we identify key integration patterns, highlight foundational challenges related to scalability, security, and ethics, and outline critical considerations for future research toward building robust, intelligent, and trustworthy decentralized systems with effective AI agent interactions. 

**Abstract (ZH)**: Web3技术与AI代理的融合代表了快速演变的前沿，有望重塑去中心化的生态系统。本文对Web3与AI代理的交集进行了首次也是最全面的分析，探讨了五个关键维度：概览、经济、治理、安全和信任机制。通过分析133个现有项目，我们首先构建了一个分类体系，并系统地描绘了当前市场的格局（RQ1），识别出项目分布和资本化中的不同模式。在此基础上，我们进一步探讨了四项关键整合：（1）AI代理在参与和优化去中心化金融中的作用（RQ2）；（2）AI代理对增强Web3治理机制的贡献（RQ3）；（3）AI代理通过智能漏洞检测和自动智能合约审计增强Web3安全的能力（RQ4）；以及（4）利用Web3内在的信任基础设施为AI代理操作建立可靠的操作框架（RQ5）。通过综合这些维度，我们识别出关键的整合模式，强调与可扩展性、安全性和伦理相关的基础挑战，并概述了未来研究中构建强大、智能和值得信赖的去中心化系统的关键考虑因素，这些系统具有有效的AI代理交互。 

---
# The Architecture of Trust: A Framework for AI-Augmented Real Estate Valuation in the Era of Structured Data 

**Title (ZH)**: 信任架构：结构化数据时代增强现实estate估值的AI框架 

**Authors**: Petteri Teikari, Mike Jarrell, Maryam Azh, Harri Pesola  

**Link**: [PDF](https://arxiv.org/pdf/2508.02765)  

**Abstract**: The Uniform Appraisal Dataset (UAD) 3.6's mandatory 2026 implementation transforms residential property valuation from narrative reporting to structured, machine-readable formats. This paper provides the first comprehensive analysis of this regulatory shift alongside concurrent AI advances in computer vision, natural language processing, and autonomous systems. We develop a three-layer framework for AI-augmented valuation addressing technical implementation and institutional trust requirements. Our analysis reveals how regulatory standardization converging with AI capabilities enables fundamental market restructuring with profound implications for professional practice, efficiency, and systemic risk. We make four key contributions: (1) documenting institutional failures including inter-appraiser variability and systematic biases undermining valuation reliability; (2) developing an architectural framework spanning physical data acquisition, semantic understanding, and cognitive reasoning that integrates emerging technologies while maintaining professional oversight; (3) addressing trust requirements for high-stakes financial applications including regulatory compliance, algorithmic fairness, and uncertainty quantification; (4) proposing evaluation methodologies beyond generic AI benchmarks toward domain-specific protocols. Our findings indicate successful transformation requires not merely technological sophistication but careful human-AI collaboration, creating systems that augment rather than replace professional expertise while addressing historical biases and information asymmetries in real estate markets. 

**Abstract (ZH)**: UAD 3.6强制实施2026年的统一评估数据集：从叙事报告到结构化、机器可读格式的转变及其对人工智能发展的影响研究 

---
# Towards a Manifesto for Cyber Humanities: Paradigms, Ethics, and Prospects 

**Title (ZH)**: 面向网络人文主义宣言：范式、伦理与前景 

**Authors**: Giovanni Adorni, Emanuele Bellini  

**Link**: [PDF](https://arxiv.org/pdf/2508.02760)  

**Abstract**: The accelerated evolution of digital infrastructures and algorithmic systems is reshaping how the humanities engage with knowledge and culture. Rooted in the traditions of Digital Humanities and Digital Humanism, the concept of "Cyber Humanities" proposes a critical reconfiguration of humanistic inquiry for the post-digital era. This Manifesto introduces a flexible framework that integrates ethical design, sustainable digital practices, and participatory knowledge systems grounded in human-centered approaches. By means of a Decalogue of foundational principles, the Manifesto invites the scientific community to critically examine and reimagine the algorithmic infrastructures that influence culture, creativity, and collective memory.
Rather than being a simple extension of existing practices, "Cyber Humanities" should be understood as a foundational paradigm for humanistic inquiry in a computationally mediated world.
Keywords: Cyber Humanities, Digital Humanities, Transdisciplinary Epistemology, Algorithmic Reflexivity, Human-centered AI, Ethics-by-Design, Knowledge Ecosystems, Digital Sovereignty, Cognitive Infrastructures 

**Abstract (ZH)**: 加速发展的数字基础设施与算法系统正在重塑人文科学与知识、文化的关系。基于数字人文和数字人文主义的传统，“赛博人文”概念提出了一种后数字时代的批判性重构人文 inquiry 的框架。这份宣言提出了一种灵活的框架，整合了伦理设计、可持续的数字实践以及以人为本的参与型知识系统。通过十项基本原则，宣言邀请科学界审视和重塑影响文化、创造力和集体记忆的算法基础设施。与现有实践的简单延伸不同，“赛博人文”应当被视为一个计算中介世界中人文 inquiry 的基础范式。

关键词：赛博人文，数字人文，跨学科认识论，算法反思性，以人为本的人工智能，设计伦理，知识生态系统，数字主权，认知基础设施 

---
# CTBench: Cryptocurrency Time Series Generation Benchmark 

**Title (ZH)**: CTBench: 加密货币时间序列生成基准 

**Authors**: Yihao Ang, Qiang Wang, Qiang Huang, Yifan Bao, Xinyu Xi, Anthony K. H. Tung, Chen Jin, Zhiyong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02758)  

**Abstract**: Synthetic time series are essential tools for data augmentation, stress testing, and algorithmic prototyping in quantitative finance. However, in cryptocurrency markets, characterized by 24/7 trading, extreme volatility, and rapid regime shifts, existing Time Series Generation (TSG) methods and benchmarks often fall short, jeopardizing practical utility. Most prior work (1) targets non-financial or traditional financial domains, (2) focuses narrowly on classification and forecasting while neglecting crypto-specific complexities, and (3) lacks critical financial evaluations, particularly for trading applications. To address these gaps, we introduce \textsf{CTBench}, the first comprehensive TSG benchmark tailored for the cryptocurrency domain. \textsf{CTBench} curates an open-source dataset from 452 tokens and evaluates TSG models across 13 metrics spanning 5 key dimensions: forecasting accuracy, rank fidelity, trading performance, risk assessment, and computational efficiency. A key innovation is a dual-task evaluation framework: (1) the \emph{Predictive Utility} task measures how well synthetic data preserves temporal and cross-sectional patterns for forecasting, while (2) the \emph{Statistical Arbitrage} task assesses whether reconstructed series support mean-reverting signals for trading. We benchmark eight representative models from five methodological families over four distinct market regimes, uncovering trade-offs between statistical fidelity and real-world profitability. Notably, \textsf{CTBench} offers model ranking analysis and actionable guidance for selecting and deploying TSG models in crypto analytics and strategy development. 

**Abstract (ZH)**: 合成时间序列是数据增强、压力测试和算法原型设计在量化金融中的重要工具。然而，在受24/7交易、极端波动性和快速制度转换特征影响的加密货币市场中，现有的时间序列生成（TSG）方法和基准通常不足，影响其实用性。大多数先前的工作（1）针对非金融或传统金融领域，（2）仅专注于分类和预测，而忽视了加密货币的特定复杂性，并且（3）缺乏关键的金融评价，特别是针对交易应用。为解决这些差距，我们引入了\textsf{CTBench}，这是第一个针对加密货币领域定制的全面TSG基准。\textsf{CTBench}收集了452种代币的数据集，并从五个关键维度的13个指标评估TSG模型：预测准确性、秩忠实性、交易性能、风险评估和计算效率。核心创新在于一个双任务评估框架：（1）预测效用任务衡量合成数据在保持时间序列和横截面模式方面的表现，以供预测使用；（2）统计套利任务评估重构时间序列支持均值回复信号的程度，以供交易使用。我们在四个不同的市场阶段比较了五种方法论家族中的八种代表模型，揭示了统计忠实性和实际盈利性之间的权衡。值得注意的是，\textsf{CTBench}提供了模型排名分析和选择及部署TSG模型在加密货币分析和策略开发中的实用指导。 

---
# Beyond the Wavefunction: Qualia Abstraction Language Mechanics and the Grammar of Awareness 

**Title (ZH)**: 超越波函数：质态抽象语言机理与觉知的语法 

**Authors**: Mikołaj Sienicki, Krzysztof Sienicki  

**Link**: [PDF](https://arxiv.org/pdf/2508.02755)  

**Abstract**: We propose a formal reconstruction of quantum mechanics grounded not in external mathematical abstractions, but in the structured dynamics of subjective experience. The Qualia Abstraction Language (QAL) models physical systems as evolving streams of introspective units, structured sequences of modality, shape, and functional effect, rather than as state vectors in Hilbert space. This approach reimagines core quantum concepts: superposition becomes a form of structured ambiguity; collapse is reframed as an introspective contraction; and entanglement is modeled as semantic resonance across streams of qualia. Drawing on insights from nominalist philosophy and oversight theoretic limits in AI, we argue that the observer paradox in quantum mechanics reflects not an ontological lacuna, but a linguistic one: the absence of a formal vocabulary for modeling first person structure. QAL introduces such a vocabulary, providing a morphodynamic framework that embeds the observer within the system and replaces abstract projection with endogenous transformation. We analyze the alignment of QAL with endophysical approaches, contrast it with standard interpretations of quantum theory, and explore its implications for a post Platonist, introspectively grounded physics. 

**Abstract (ZH)**: 我们提出了一种基于主观体验结构化动力学的量子力学形式重构，而非基于外部数学抽象。质态抽象语言（QAL）将物理系统建模为内省单元的演化流，而不是希拉勃空间中的状态矢量，其中这些单元以模态、形状和功能效果的结构化序列形式存在。这种方法重新构想了核心量子概念：超position转变为结构化的模态模糊性；坍缩重新定义为内省收缩；纠缠则被建模为隔断不同质态流之间的语义共振。借鉴名义主义哲学和AI元监督理论的限制，我们认为量子力学中的观察者悖论反映的不是本体论上的缺失，而是语言上的缺失：缺乏一种形式化的语言来建模第一人称结构。QAL引入了这种语言，提供了一种形态动力学框架，将观察者嵌入系统中，并用内生转化替代了抽象投影。我们分析了QAL与端体物理方法的一致性，将其与量子理论的标准解释进行了对比，并探讨了其对未来以内省为基础物理学的启示。 

---
# DMSC: Dynamic Multi-Scale Coordination Framework for Time Series Forecasting 

**Title (ZH)**: DMSC：动态多尺度协调框架用于时间序列预测 

**Authors**: Haonan Yang, Jianchao Tang, Zhuo Li, Long Lan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02753)  

**Abstract**: Time Series Forecasting (TSF) faces persistent challenges in modeling intricate temporal dependencies across different scales. Despite recent advances leveraging different decomposition operations and novel architectures based on CNN, MLP or Transformer, existing methods still struggle with static decomposition strategies, fragmented dependency modeling, and inflexible fusion mechanisms, limiting their ability to model intricate temporal dependencies. To explicitly solve the mentioned three problems respectively, we propose a novel Dynamic Multi-Scale Coordination Framework (DMSC) with Multi-Scale Patch Decomposition block (EMPD), Triad Interaction Block (TIB) and Adaptive Scale Routing MoE block (ASR-MoE). Specifically, EMPD is designed as a built-in component to dynamically segment sequences into hierarchical patches with exponentially scaled granularities, eliminating predefined scale constraints through input-adaptive patch adjustment. TIB then jointly models intra-patch, inter-patch, and cross-variable dependencies within each layer's decomposed representations. EMPD and TIB are jointly integrated into layers forming a multi-layer progressive cascade architecture, where coarse-grained representations from earlier layers adaptively guide fine-grained feature extraction in subsequent layers via gated pathways. And ASR-MoE dynamically fuses multi-scale predictions by leveraging specialized global and local experts with temporal-aware weighting. Comprehensive experiments on thirteen real-world benchmarks demonstrate that DMSC consistently maintains state-of-the-art (SOTA) performance and superior computational efficiency for TSF tasks. Code is available at this https URL. 

**Abstract (ZH)**: 动态多尺度协调框架在时间序列 forecasting 中的多尺度协调预测 

---
# Pulse Shape Discrimination Algorithms: Survey and Benchmark 

**Title (ZH)**: 脉冲形状鉴别算法：综述与基准 

**Authors**: Haoran Liu, Yihan Zhan, Mingzhe Liu, Yanhua Liu, Peng Li, Zhuo Zuo, Bingqi Liu, Runxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02750)  

**Abstract**: This review presents a comprehensive survey and benchmark of pulse shape discrimination (PSD) algorithms for radiation detection, classifying nearly sixty methods into statistical (time-domain, frequency-domain, neural network-based) and prior-knowledge (machine learning, deep learning) paradigms. We implement and evaluate all algorithms on two standardized datasets: an unlabeled set from a 241Am-9Be source and a time-of-flight labeled set from a 238Pu-9Be source, using metrics including Figure of Merit (FOM), F1-score, ROC-AUC, and inter-method correlations. Our analysis reveals that deep learning models, particularly Multi-Layer Perceptrons (MLPs) and hybrid approaches combining statistical features with neural regression, often outperform traditional methods. We discuss architectural suitabilities, the limitations of FOM, alternative evaluation metrics, and performance across energy thresholds. Accompanying this work, we release an open-source toolbox in Python and MATLAB, along with the datasets, to promote reproducibility and advance PSD research. 

**Abstract (ZH)**: 这篇综述对辐射检测中脉冲形状鉴别（PSD）算法进行了全面的调查和基准测试，将近六十种方法分类为统计方法（时域、频域、基于神经网络）和先验知识方法（机器学习、深度学习）。我们在两个标准化数据集上实现并评估了所有算法：来自241Am-9Be源的未标记数据集和来自238Pu-9Be源的时间飞行标记数据集，使用的评估指标包括效能指标（FOM）、F1分数、ROC-AUC以及方法间相关性。我们的分析表明，深度学习模型，尤其是多层感知器（MLPs）和结合统计特征与神经回归的混合方法，往往优于传统方法。我们讨论了架构适用性、效能指标FOM的局限性、替代评估指标以及不同能量阈值下的性能。为了促进可重复性和推进PSD研究，我们还发布了基于Python和MATLAB的开源工具包，以及相关数据集。 

---
# A Novel cVAE-Augmented Deep Learning Framework for Pan-Cancer RNA-Seq Classification 

**Title (ZH)**: 一种新型cVAE增强增强深度学习框架，用于泛癌RNA-se-Seq分类 

**Authors**: Vinil Polepalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.02743)  

**Abstract**: Pan-cancer classification using transcriptomic (RNA-Seq) data can inform tumor subtyping and therapy selection, but is challenging due to extremely high dimensionality and limited sample sizes. In this study, we propose a novel deep learning framework that uses a class-conditional variational autoencoder (cVAE) to augment training data for pan-cancer gene expression classification. Using 801 tumor RNA-Seq samples spanning 5 cancer types from The Cancer Genome Atlas (TCGA), we first perform feature selection to reduce 20,531 gene expression features to the 500 most variably expressed genes. A cVAE is then trained on this data to learn a latent representation of gene expression conditioned on cancer type, enabling the generation of synthetic gene expression samples for each tumor class. We augment the training set with these cVAE-generated samples (doubling the dataset size) to mitigate overfitting and class imbalance. A two-layer multilayer perceptron (MLP) classifier is subsequently trained on the augmented dataset to predict tumor type. The augmented framework achieves high classification accuracy (~98%) on a held-out test set, substantially outperforming a classifier trained on the original data alone. We present detailed experimental results, including VAE training curves, classifier performance metrics (ROC curves and confusion matrix), and architecture diagrams to illustrate the approach. The results demonstrate that cVAE-based synthetic augmentation can significantly improve pan-cancer prediction performance, especially for underrepresented cancer classes. 

**Abstract (ZH)**: 使用转录组（RNA-Seq）数据进行泛癌种分类可以指导肿瘤亚型划分和治疗选择，但由于数据的极高维度和样本量有限，这一过程具有挑战性。在本研究中，我们提出了一种新的深度学习框架，使用类别条件变分自编码器（cVAE）扩充泛癌种基因表达分类的训练数据。使用来自《癌基因组数据库》（TCGA）的801个跨越5种癌症类型的肿瘤RNA-Seq样本，我们首先进行特征选择，将20,531个基因表达特征减少到500个最可变表达的基因。然后，我们在该数据上训练一个cVAE，以在条件于癌症类型的基因表达中学习一个潜在表示，使我们能够为每个肿瘤类别生成合成基因表达样本。我们将这些cVAE生成的样本添加到训练集中（使数据集大小加倍），以减轻过拟合和类别不平衡问题。随后，在扩充的数据集上训练一个两层多层感知器（MLP）分类器，以预测肿瘤类型。扩充框架在保留的数据集上实现了高分类准确率（约98%），明显优于仅在原始数据上训练的分类器。我们详细呈现了实验结果，包括VAE训练曲线、分类器性能指标（ROC曲线和混淆矩阵）以及架构图，以说明本方法。结果表明，基于cVAE的合成扩充可以显著提高泛癌种预测性能，尤其是在稀有癌种类别的预测上。 

---
# SpectrumFM: A New Paradigm for Spectrum Cognition 

**Title (ZH)**: 频谱FM：一种新的频谱认知 paradigm 

**Authors**: Chunyu Liu, Hao Zhang, Wei Wu, Fuhui Zhou, Qihui Wu, Derrick Wing Kwan Ng, Chan-Byoung Chae  

**Link**: [PDF](https://arxiv.org/pdf/2508.02742)  

**Abstract**: The enhancement of spectrum efficiency and the realization of secure spectrum utilization are critically dependent on spectrum cognition. However, existing spectrum cognition methods often exhibit limited generalization and suboptimal accuracy when deployed across diverse spectrum environments and tasks. To overcome these challenges, we propose a spectrum foundation model, termed SpectrumFM, which provides a new paradigm for spectrum cognition. An innovative spectrum encoder that exploits the convolutional neural networks and the multi-head self attention mechanisms is proposed to effectively capture both fine-grained local signal structures and high-level global dependencies in the spectrum data. To enhance its adaptability, two novel self-supervised learning tasks, namely masked reconstruction and next-slot signal prediction, are developed for pre-training SpectrumFM, enabling the model to learn rich and transferable representations. Furthermore, low-rank adaptation (LoRA) parameter-efficient fine-tuning is exploited to enable SpectrumFM to seamlessly adapt to various downstream spectrum cognition tasks, including spectrum sensing (SS), anomaly detection (AD), and wireless technology classification (WTC). Extensive experiments demonstrate the superiority of SpectrumFM over state-of-the-art methods. Specifically, it improves detection probability in the SS task by 30% at -4 dB signal-to-noise ratio (SNR), boosts the area under the curve (AUC) in the AD task by over 10%, and enhances WTC accuracy by 9.6%. 

**Abstract (ZH)**: 基于SpectrumFM的谱认知基础模型：提升频谱效率与实现安全频谱利用的新范式 

---
# DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening 

**Title (ZH)**: DeepGB-TB：一种快速可解释的平衡风险交叉注意力梯度提升卷积网络结核病筛查方法 

**Authors**: Zhixiang Lu, Yulong Li, Feilong Tang, Zhengyong Jiang, Chong Li, Mian Zhou, Tenglong Li, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.02741)  

**Abstract**: Large-scale tuberculosis (TB) screening is limited by the high cost and operational complexity of traditional diagnostics, creating a need for artificial-intelligence solutions. We propose DeepGB-TB, a non-invasive system that instantly assigns TB risk scores using only cough audio and basic demographic data. The model couples a lightweight one-dimensional convolutional neural network for audio processing with a gradient-boosted decision tree for tabular features. Its principal innovation is a Cross-Modal Bidirectional Cross-Attention module (CM-BCA) that iteratively exchanges salient cues between modalities, emulating the way clinicians integrate symptoms and risk factors. To meet the clinical priority of minimizing missed cases, we design a Tuberculosis Risk-Balanced Loss (TRBL) that places stronger penalties on false-negative predictions, thereby reducing high-risk misclassifications. DeepGB-TB is evaluated on a diverse dataset of 1,105 patients collected across seven countries, achieving an AUROC of 0.903 and an F1-score of 0.851, representing a new state of the art. Its computational efficiency enables real-time, offline inference directly on common mobile devices, making it ideal for low-resource settings. Importantly, the system produces clinically validated explanations that promote trust and adoption by frontline health workers. By coupling AI innovation with public-health requirements for speed, affordability, and reliability, DeepGB-TB offers a tool for advancing global TB control. 

**Abstract (ZH)**: 大型规模结核病（TB）筛查受限于传统诊断方法的高成本和操作复杂性，亟需人工智能解决方案。我们提出DeepGB-TB，这是一种非侵入性系统，仅使用咳嗽声音和基本人口统计数据即可瞬间分配结核病风险评分。该模型结合了轻量级的一维卷积神经网络进行音频处理，并结合梯度提升决策树处理表格特征。其主要创新在于跨模态双向交叉注意模块（CM-BCA），该模块在不同模态之间迭代交换关键线索，模仿临床医生整合症状和风险因素的方式。为符合临床优先级，即尽可能减少漏诊，我们设计了一种结核病风险平衡损失（TRBL），加重了假阴性预测的惩罚，从而减少高风险的误分类。DeepGB-TB在来自七个国家的1,105名患者的多样化数据集上进行评估，AUROC为0.903，F1分数为0.851，代表了新的技术水平。其计算效率使其能够在常见移动设备上实现实时、离线推理，适用于资源匮乏的地区。更重要的是，该系统生成了临床验证的解释，有助于前线卫生工作者的信任和采用。通过结合AI创新与公共卫生对速度、成本效益和可靠性的要求，DeepGB-TB提供了一种推进全球结核病控制的工具。 

---
# Interpreting Performance Profiles with Deep Learning 

**Title (ZH)**: 基于深度学习的性能概貌解释 

**Authors**: Zhuoran Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02729)  

**Abstract**: Profiling tools (also known as profilers) play an important role in understanding program performance at runtime, such as hotspots, bottlenecks, and inefficiencies. While profilers have been proven to be useful, they give extra burden to software engineers. Software engineers, as the users, are responsible to interpret the complex performance data and identify actionable optimization in program source code. However, it can be challenging for users to associate inefficiencies with the program semantics, especially if the users are not the authors of the code, which limits the applicability of profilers.
In this thesis, we explore a new direction to combine performance profiles and program semantics with a deep learning approach. The key idea is to glean code summary for semantic information (at a certain level) and integrate it into a profiler, which can better understand program inefficiencies for actionable optimization. To be concrete, we combine profiles generated by Async Profiler (the state-of-the-art Java profiler) with code summarization from a fine-tuned CodeBERT-based model. We demonstrate the code summaries of any selected call path in a graphic user interface. Our system can effectively assist analysis on many Java benchmarks. 

**Abstract (ZH)**: 性能分析工具与程序语义结合的新方向：基于深度学习的方法 

---
# Forecasting NCAA Basketball Outcomes with Deep Learning: A Comparative Study of LSTM and Transformer Models 

**Title (ZH)**: 使用深度学习预测NCAA篮球比赛结果：LSTM和Transformer模型的比较研究 

**Authors**: Md Imtiaz Habib  

**Link**: [PDF](https://arxiv.org/pdf/2508.02725)  

**Abstract**: In this research, I explore advanced deep learning methodologies to forecast the outcomes of the 2025 NCAA Division 1 Men's and Women's Basketball tournaments. Leveraging historical NCAA game data, I implement two sophisticated sequence-based models: Long Short-Term Memory (LSTM) and Transformer architectures. The predictive power of these models is augmented through comprehensive feature engineering, including team quality metrics derived from Generalized Linear Models (GLM), Elo ratings, seed differences, and aggregated box-score statistics. To evaluate the robustness and reliability of predictions, I train each model variant using both Binary Cross-Entropy (BCE) and Brier loss functions, providing insights into classification performance and probability calibration. My comparative analysis reveals that while the Transformer architecture optimized with BCE yields superior discriminative power (highest AUC of 0.8473), the LSTM model trained with Brier loss demonstrates superior probabilistic calibration (lowest Brier score of 0.1589). These findings underscore the importance of selecting appropriate model architectures and loss functions based on the specific requirements of forecasting tasks. The detailed analytical pipeline presented here serves as a reproducible framework for future predictive modeling tasks in sports analytics and beyond. 

**Abstract (ZH)**: 本研究探索先进的深度学习方法以预测2025年NCAA Division 1男子和女子篮球锦标赛的 outcomes。利用历史NCAA比赛数据，我实现了两种复杂的序列模型：长短期记忆（LSTM）和变压器架构。通过全面的特征工程，包括从广义线性模型（GLM）导出的团队质量指标、Elo排名、种子差异和综合比赛统计，提升了这些模型的预测能力。通过使用二元交叉熵（BCE）和贝列尔散度（Brier loss）这两种损失函数训练每个模型变体，评估了预测的稳健性和可靠性，并为分类性能和概率校准提供了见解。比较分析表明，在使用二元交叉熵优化的变压器架构在辨别能力（最高AUC为0.8473）方面表现出色，而使用贝列尔散度训练的LSTM模型在概率校准（最低贝列尔分数为0.1589）方面表现更优。这些发现强调了根据预测任务的具体要求选择合适的模型架构和损失函数的重要性。本文详细的分析管道提供了一个可重复的框架，用于未来体育分析中的预测建模任务及其他领域。 

---
# Veli: Unsupervised Method and Unified Benchmark for Low-Cost Air Quality Sensor Correction 

**Title (ZH)**: Veli：一种无监督方法及统一基准用于低成本空气质量传感器校正 

**Authors**: Yahia Dalbah, Marcel Worring, Yen-Chia Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02724)  

**Abstract**: Urban air pollution is a major health crisis causing millions of premature deaths annually, underscoring the urgent need for accurate and scalable monitoring of air quality (AQ). While low-cost sensors (LCS) offer a scalable alternative to expensive reference-grade stations, their readings are affected by drift, calibration errors, and environmental interference. To address these challenges, we introduce Veli (Reference-free Variational Estimation via Latent Inference), an unsupervised Bayesian model that leverages variational inference to correct LCS readings without requiring co-location with reference stations, eliminating a major deployment barrier. Specifically, Veli constructs a disentangled representation of the LCS readings, effectively separating the true pollutant reading from the sensor noise. To build our model and address the lack of standardized benchmarks in AQ monitoring, we also introduce the Air Quality Sensor Data Repository (AQ-SDR). AQ-SDR is the largest AQ sensor benchmark to date, with readings from 23,737 LCS and reference stations across multiple regions. Veli demonstrates strong generalization across both in-distribution and out-of-distribution settings, effectively handling sensor drift and erratic sensor behavior. Code for model and dataset will be made public when this paper is published. 

**Abstract (ZH)**: 城市空气污染是每年造成数百万人早死的重大健康危机，突显了准确和可扩展的空气质量监测的迫切需求。虽然低成本传感器提供了昂贵标准站的可扩展替代方案，但其读数受到漂移、校准误差和环境干扰的影响。为了解决这些挑战，我们提出了Veli（基于潜在推理的无监督变分估计），一个利用变分推断的无监督贝叶斯模型，无需与参考站共址即可校正低成本传感器读数，消除了主要部署障碍。具体而言，Veli 构建了低成本传感器读数的分解表示，有效地将真实的污染物读数与传感器噪声分离。为构建我们的模型并应对空气质量监测中缺乏标准化基准的问题，我们还引入了空气质量传感器数据仓库（AQ-SDR）。AQ-SDR 是迄今为止最大的空气质量传感器基准，包含来自多个地区共计23,737个低成本传感器和参考站的数据。Veli 在分布内和分布外场景下均表现出强大的泛化能力，有效处理了传感器漂移和异常传感器行为。该论文发表时，模型代码和数据集将开源。 

---
# Mathematical Foundations of Geometric Deep Learning 

**Title (ZH)**: 几何深度学习的数学基础 

**Authors**: Haitz Sáez de Ocáriz Borde, Michael Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.02723)  

**Abstract**: We review the key mathematical concepts necessary for studying Geometric Deep Learning. 

**Abstract (ZH)**: 我们回顾研究几何深度学习所需的key数学概念。 

---
# ECGTwin: Personalized ECG Generation Using Controllable Diffusion Model 

**Title (ZH)**: ECGTwin：基于可控扩散模型的个性化心电图生成 

**Authors**: Yongfan Lai, Bo Liu, Xinyan Guan, Qinghao Zhao, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02720)  

**Abstract**: Personalized electrocardiogram (ECG) generation is to simulate a patient's ECG digital twins tailored to specific conditions. It has the potential to transform traditional healthcare into a more accurate individualized paradigm, while preserving the key benefits of conventional population-level ECG synthesis. However, this promising task presents two fundamental challenges: extracting individual features without ground truth and injecting various types of conditions without confusing generative model. In this paper, we present ECGTwin, a two-stage framework designed to address these challenges. In the first stage, an Individual Base Extractor trained via contrastive learning robustly captures personal features from a reference ECG. In the second stage, the extracted individual features, along with a target cardiac condition, are integrated into the diffusion-based generation process through our novel AdaX Condition Injector, which injects these signals via two dedicated and specialized pathways. Both qualitative and quantitative experiments have demonstrated that our model can not only generate ECG signals of high fidelity and diversity by offering a fine-grained generation controllability, but also preserving individual-specific features. Furthermore, ECGTwin shows the potential to enhance ECG auto-diagnosis in downstream application, confirming the possibility of precise personalized healthcare solutions. 

**Abstract (ZH)**: 个性化心电图（ECG）生成是模拟特定条件下患者的心电图数字双胞胎。它有可能将传统医疗转变为更加准确的个性化范式，同时保留传统群体级心电图合成的关键优势。然而，这一有前景的任务面临着两大基本挑战：在缺乏真实标注的情况下提取个体特征，以及在不混淆生成模型的情况下注入各种类型的心脏状况。在本文中，我们提出了ECGTwin，这是一种两阶段框架，旨在解决这些问题。在第一阶段，通过对比学习训练的个体基底提取器稳健地从参考心电图中捕获个人特征。在第二阶段，通过我们新型的AdaX条件注入器提取的个体特征与目标心脏状况结合，通过两条专门且专门化的路径将这些信号集成到基于扩散的生成过程中。实验证明，我们的模型不仅能够通过提供细腻的生成可控性生成高质量、多样性的ECG信号，还能保留个体特有的特征。此外，ECGTwin展示了在下游应用中增强心电图自动诊断的潜力，证明了精准个性化医疗解决方案的可能性。 

---
# ZetA: A Riemann Zeta-Scaled Extension of Adam for Deep Learning 

**Title (ZH)**: ZetA：基于Riemann Zeta尺度扩展的Adam算法用于深度学习 kukuk 

**Authors**: Samiksha BC  

**Link**: [PDF](https://arxiv.org/pdf/2508.02719)  

**Abstract**: This work introduces ZetA, a novel deep learning optimizer that extends Adam by incorporating dynamic scaling based on the Riemann zeta function. To the best of our knowledge, ZetA is the first optimizer to apply zeta-based gradient scaling within deep learning optimization. The method improves generalization and robustness through a hybrid update mechanism that integrates adaptive damping, cosine similarity-based momentum boosting, entropy-regularized loss, and Sharpness-Aware Minimization (SAM)-style perturbations. Empirical evaluations on SVHN, CIFAR10, CIFAR100, STL10, and noisy CIFAR10 consistently show test accuracy improvements over Adam. All experiments employ a lightweight fully connected network trained for five epochs under mixed-precision settings. The results demonstrate that ZetA is a computationally efficient and robust alternative to Adam, particularly effective in noisy or high-granularity classification tasks. 

**Abstract (ZH)**: 基于黎曼ζ函数的动态缩放的新型深度学习优化器ZetA 

---
# SleepLiteCNN: Energy-Efficient Sleep Apnea Subtype Classification with 1-Second Resolution Using Single-Lead ECG 

**Title (ZH)**: SleepLiteCNN：基于单导联心电图的高效睡眠呼吸暂停亚型分类方法，分辨率为1秒 

**Authors**: Zahra Mohammadi, Siamak Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02718)  

**Abstract**: Apnea is a common sleep disorder characterized by breathing interruptions lasting at least ten seconds and occurring more than five times per hour. Accurate, high-temporal-resolution detection of sleep apnea subtypes - Obstructive, Central, and Mixed - is crucial for effective treatment and management. This paper presents an energy-efficient method for classifying these subtypes using a single-lead electrocardiogram (ECG) with high temporal resolution to address the real-time needs of wearable devices. We evaluate a wide range of classical machine learning algorithms and deep learning architectures on 1-second ECG windows, comparing their accuracy, complexity, and energy consumption. Based on this analysis, we introduce SleepLiteCNN, a compact and energy-efficient convolutional neural network specifically designed for wearable platforms. SleepLiteCNN achieves over 95% accuracy and a 92% macro-F1 score, while requiring just 1.8 microjoules per inference after 8-bit quantization. Field Programmable Gate Array (FPGA) synthesis further demonstrates significant reductions in hardware resource usage, confirming its suitability for continuous, real-time monitoring in energy-constrained environments. These results establish SleepLiteCNN as a practical and effective solution for wearable device sleep apnea subtype detection. 

**Abstract (ZH)**: 无眠症是一种常见的睡眠障碍，其特征是呼吸中断持续至少10秒，且每小时发生超过5次。准确、高时间分辨率地检测无眠症亚型——阻塞性、中枢性及混合性——对于有效的治疗和管理至关重要。本文提出了一种节能方法，利用高时间分辨率的一导联心电图（ECG）对这些亚型进行分类，以满足穿戴设备的实时需求。我们评估了多种经典的机器学习算法和深度学习架构在1秒心电信号窗口上的性能，比较了它们的准确度、复杂度和能耗。在此基础上，我们引入了SleepLiteCNN，这是一种专为穿戴平台设计的紧凑且节能的卷积神经网络。SleepLiteCNN在8位量化后实现超过95%的准确率和92%的宏F1分数，且每推断一次仅需1.8微焦耳能量。现场可编程门阵列（FPGA）综合进一步证明了其在硬件资源使用方面的显著减少，确认其适用于能量受限环境下的连续、实时监测。这些结果确立了SleepLiteCNN作为穿戴设备无眠症亚型检测的一种实用且有效的解决方案。 

---
# Evaluation of Deep Learning Models for LBBB Classification in ECG Signals 

**Title (ZH)**: 深度学习模型在心电图信号LBBB分类中的评估 

**Authors**: Beatriz Macas Ordóñez, Diego Vinicio Orellana Villavicencio, José Manuel Ferrández, Paula Bonomini  

**Link**: [PDF](https://arxiv.org/pdf/2508.02710)  

**Abstract**: This study explores different neural network architectures to evaluate their ability to extract spatial and temporal patterns from electrocardiographic (ECG) signals and classify them into three groups: healthy subjects, Left Bundle Branch Block (LBBB), and Strict Left Bundle Branch Block (sLBBB).
Clinical Relevance, Innovative technologies enable the selection of candidates for Cardiac Resynchronization Therapy (CRT) by optimizing the classification of subjects with Left Bundle Branch Block (LBBB). 

**Abstract (ZH)**: 本研究探讨不同的神经网络架构，评估其从心电图（ECG）信号中提取空间和时间模式的能力，并将信号分类为三组：健康受试者、左束支阻滞（LBBB）和严格左束支阻滞（sLBBB）。

临床相关性：创新技术可通过优化左束支阻滞（LBBB）患者分类，选择心脏再同步化治疗（CRT）的候选者。 

---
# AnnoSense: A Framework for Physiological Emotion Data Collection in Everyday Settings for AI 

**Title (ZH)**: AnnoSense: 一种在日常场景中收集生理情感数据的框架用于AI 

**Authors**: Pragya Singh, Ankush Gupta, Mohan Kumar, Pushpendra Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.02680)  

**Abstract**: Emotional and mental well-being are vital components of quality of life, and with the rise of smart devices like smartphones, wearables, and artificial intelligence (AI), new opportunities for monitoring emotions in everyday settings have emerged. However, for AI algorithms to be effective, they require high-quality data and accurate annotations. As the focus shifts towards collecting emotion data in real-world environments to capture more authentic emotional experiences, the process of gathering emotion annotations has become increasingly complex. This work explores the challenges of everyday emotion data collection from the perspectives of key stakeholders. We collected 75 survey responses, performed 32 interviews with the public, and 3 focus group discussions (FGDs) with 12 mental health professionals. The insights gained from a total of 119 stakeholders informed the development of our framework, AnnoSense, designed to support everyday emotion data collection for AI. This framework was then evaluated by 25 emotion AI experts for its clarity, usefulness, and adaptability. Lastly, we discuss the potential next steps and implications of AnnoSense for future research in emotion AI, highlighting its potential to enhance the collection and analysis of emotion data in real-world contexts. 

**Abstract (ZH)**: 智能设备兴起背景下情绪和心理福祉对于生活质量的重要性及其数据收集挑战：AnnoSense框架的开发与评估 

---
